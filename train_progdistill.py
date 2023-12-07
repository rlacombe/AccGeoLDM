import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
from equivariant_diffusion import utils as diffusion_utils


def train_epoch(args, loader, epoch, teacher, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    model_dp.train()
    model.train() # NOTE 'model' is the student model (takes $args.diffusion_steps steps), 'teacher' is the teacher
    loss_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # Compute teacher target (no_gard so the teacher stays fixed)
        with torch.no_grad():

            # Send batch to latent space
            z_x, z_h =  encode_to_latent_space(teacher, x, h, node_mask, edge_mask, context) 

            # Sample time steps
            t, u, v, alpha_t, sigma_t, alpha_u, sigma_u, alpha_v, sigma_v = sample_time_steps(teacher, args.diffusion_steps, z_x)

            # Sample zt ~ Normal(alpha_t x, sigma_t)
            eps = teacher.sample_combined_position_feature_noise(n_samples=x.size(0), 
                                                                 n_nodes=x.size(1), node_mask=node_mask)

            # Concatenate x, h[integer] and h[categorical].
            xh = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)

            # Sample z_t given x, h for timestep t, from q(z_t | x, h)
            z_t = alpha_t * xh + sigma_t * eps

            diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :model.n_dims], node_mask)

            # Compute double denoising steps
            xhat_zt = denoise_step(teacher, z_t, alpha_t, sigma_t, t, node_mask, edge_mask, context)
            z_u = alpha_u * xhat_zt + (sigma_u/sigma_t) * (z_t - alpha_t * xhat_zt)

            xhat_zu = denoise_step(teacher, z_u, alpha_u, sigma_u, u, node_mask, edge_mask, context)
            z_v = alpha_v * xhat_zu + (sigma_v/sigma_u) * (z_u - alpha_u * xhat_zu)

            teacher_target = (z_v - (sigma_v/sigma_t)*z_t)/(alpha_v - (sigma_v/sigma_t)*alpha_t)

        # Detach target and inputs for *extra* caution
        teacher_target.detach()
        z_t.detach()
        alpha_t.detach()
        sigma_t.detach()
        t.detach()        

        # Foward pass of the student
        student_target = denoise_step(model, z_t, alpha_t, sigma_t, t, node_mask, edge_mask, context)
        
        # Compute loss
        loss = torch.square(student_target - teacher_target)
        loss = loss.mean()

        # Take backward pass
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        # Optimize student params
        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}"
                  f"GradNorm: {grad_norm:.1f}")
        loss_epoch.append(loss.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0) and args.train_diffusion:
            start = time.time()
            if len(args.conditioning) > 0:
                save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                  batch_id=str(i))
            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            if len(args.conditioning) > 0:
                vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
                                    wandb=wandb, mode='conditional')
        wandb.log({"Batch Loss": loss.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch Loss": np.mean(loss_epoch)}, commit=False)



def sample_time_steps(model, N, x):

        # Sample a timestep t.
        t_int = torch.randint(
            1, model.T + 1, size=(x.size(0), 1), device=x.device).float()

        t = t_int / model.T
        u = t - .5/N
        v = t - 1/N

        # Compute gamma_s and gamma_t via the network.
        gamma_t = model.inflate_batch_array(model.gamma(t), x)
        gamma_u = model.inflate_batch_array(model.gamma(u), x)
        gamma_v = model.inflate_batch_array(model.gamma(v), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t, sigma_t = model.alpha(gamma_t, x), model.sigma(gamma_t, x)
        alpha_u, sigma_u = model.alpha(gamma_u, x), model.sigma(gamma_u, x)
        alpha_v, sigma_v = model.alpha(gamma_v, x), model.sigma(gamma_v, x)

        return t, u, v, alpha_t, sigma_t, alpha_u, sigma_u, alpha_v, sigma_v


def encode_to_latent_space(model, x, h, node_mask, edge_mask, context):

        # Encode data to latent space.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = model.vae.encode(x, h, node_mask, edge_mask, context)
        # Compute fixed sigma values.
        t_zeros = torch.zeros(size=(x.size(0), 1), device = x.device)
        model.gamma.to('cuda')
        gamma_0 = model.inflate_batch_array(model.gamma(t_zeros), x)
        sigma_0 = model.sigma(gamma_0, x)

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = sigma_0
        # z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = model.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        z_xh = z_xh.detach()  # Always keep the encoder fixed.
        diffusion_utils.assert_correctly_masked(z_xh, node_mask)

        z_x = z_xh[:, :, :model.n_dims]
        z_h = z_xh[:, :, model.n_dims:]
        diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}

        return z_x, z_h


def denoise_step(model, z_t, alpha_t, sigma_t, t, node_mask, edge_mask, context):
        
    return z_t / alpha_t - model.phi(z_t, t, node_mask, edge_mask, context) * sigma_t / alpha_t
        
        

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        loss_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            loss, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            # standard nll from forward KL

            loss_epoch += loss.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {loss_epoch/n_samples:.2f}")

    return loss_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=10, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
