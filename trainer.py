import numpy as np
import os.path
import torch

from models import get_c_loss, loss_function
from utils import reset_grad, from_numpy_to_var
from logs import log_info, log_images
from dataset import get_torch_images_from_numpy, get_idx_t, get_torch_actions, get_negative_examples
from tensorboard_logger import configure
import tqdm

def train(all_models, training_models, solver, training_params, log_every, **kwargs):
    model, c_model, actor = all_models
    k_steps = kwargs["k"]
    num_epochs = kwargs["n_epochs"]
    batch_size = kwargs["batch_size"]
    N = kwargs["N"]
    c_type = kwargs["c_type"]
    vae_weight = kwargs["vae_w"]
    beta = kwargs["vae_b"]

    # Configure experiment path
    savepath = kwargs['savepath']

    conditional = kwargs["conditional"]

    configure('%s/var_log' % savepath, flush_secs=5)

    ### Load data ### -- assuming appropriate npy format
    data_file = kwargs["data_dir"]
    data = np.load(data_file,allow_pickle=True)
    meta = data.item().get('meta')
    path_len = meta.item().get('path_len')
    n_trajs = meta.item().get('num_paths')
    data = data.item().get('top_views')
    coords = data.item().get('coordinates')
    data_size = path_len * n_trajs
    if len(data.shape) == 4:
        data = data.reshape(n_trajs,path_len,*data.shape[1:])
    print('Number of trajectories: %d' % n_trajs)  # 315
    print('Path length: %d' % path_len)  # 315
    print('Number of transitions: %d' % data_size)  # 378315

    ### Train models ###
    c_loss = vae_loss = a_loss = torch.Tensor([0]).cuda()
    for epoch in range(num_epochs):
        n_batch = 10 #int(data_size / batch_size)
        print('********** Epoch %i ************' % epoch)
        for it in tqdm.tqdm(range(n_batch)):
            idx, t = get_idx_t(batch_size, k_steps, path_len, n_trajs, data)
            o, c = get_torch_images_from_numpy(data[idx, t], conditional)
            ks = np.random.choice(k_steps, batch_size)
            o_next, _ = get_torch_images_from_numpy(data[idx, t + ks], conditional)
            o_neg = get_negative_examples(data, idx, batch_size, N, n_trajs,path_len,coords) 
            
            o_pred, mu, logvar, cond_info = model(o, c)
            o_next_pred, _, _, _ = model(o_next, c)

            # VAE loss
            if model in training_models:
                vae_loss = loss_function(o_pred, o, mu, logvar,
                                         cond_info.get("means_cond", None),
                                         cond_info.get("log_var_cond", None),
                                         beta=beta) * vae_weight
                vae_loss.backward()

            # C loss
            if c_model in training_models and epoch >= kwargs["pretrain"]:
                c_loss = get_c_loss(model,
                                    c_model,
                                    c_type,
                                    o_pred,
                                    o_next_pred,
                                    c,
                                    N,
                                    o_neg)
                c_loss.backward()

            # Actor loss
            if actor in training_models and epoch >= kwargs["pretrain"]:
                a = get_torch_actions(data[idx, t + 1])
                a_loss = actor.loss(a, o, o_next, c)
                a_loss.backward()

            ### Update models ###
            if solver is not None:
                solver.step()
            reset_grad(training_params)

            if it % log_every == 0:
                ### Log info ###
                log_info(c_loss, vae_loss, a_loss, model,
                         conditional, cond_info,
                         it, n_batch, epoch)

                ### Save params ###
                if not os.path.exists('%s/var' % savepath):
                    os.makedirs('%s/var' % savepath)
                n_models = 1
                torch.save(model.state_dict(), '%s/var/vae-%d-last-5' % (savepath, epoch % n_models + 1 ))
                torch.save(c_model.state_dict(), '%s/var/cpc-%d-last-5' % (savepath, epoch % n_models + 1 ))
                torch.save(actor.state_dict(), '%s/var/actor-%d-last-5' % (savepath, epoch % n_models + 1 ))
