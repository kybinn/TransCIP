import os
import torch
import numpy as np
from models import extractionForCancer,extractionForCCC,BipartiteCCC,Discriminator,Decoder,ReverseLayerF
import config
from itertools import chain
from itertools import cycle
from torch.nn import functional as F
from utils.loss_functions import Adversarial_loss,instanceloss,coral,ortho_loss
from torch.utils.tensorboard import SummaryWriter


def eval_epoch(lam1,cancer_encoder, ccc_encoder,ccc_gate_layer,decoder_cancer, discriminator1,loader1, loader2, device):
    total_loss = 0

    for i, batch in enumerate(zip(loader2, cycle(loader1))):
        s_batch=batch[1]
        t_batch=batch[0]
        s_gene_all = s_batch[0].to(device)    # torch.Size([128, 1306])
        s_ccc = s_batch[1].to(device) # torch.Size([128, 732])
        t_gene_all = t_batch[0].to(device)
        t_ccc = t_batch[1].to(device)
                

        with torch.no_grad():
            s_cancer = cancer_encoder(s_gene_all)                 # torch.Size([128, 64])
            s_ccc_embed = ccc_encoder(s_ccc)
            s_ccc_gate = torch.sigmoid(ccc_gate_layer(s_ccc_embed))
            s_contextual_gene = s_cancer * s_ccc_gate
            s_domain_output1 = discriminator1(s_contextual_gene)             # torch.Size([128, 1])
            s_fea_rec1 = decoder_cancer(s_contextual_gene)                        # torch.Size([128, ??])

            t_cancer = cancer_encoder(t_gene_all)
            t_ccc_embed = ccc_encoder(t_ccc)
            t_ccc_gate = torch.sigmoid(ccc_gate_layer(t_ccc_embed))
            t_contextual_gene = t_cancer * t_ccc_gate
            t_domain_output1 = discriminator1(t_contextual_gene)
            t_fea_rec1 = decoder_cancer(t_contextual_gene)

    
            # reconstruct loss
            rec_s = F.mse_loss(s_gene_all, s_fea_rec1)
            rec_t = F.mse_loss(t_gene_all, t_fea_rec1)
            recon_loss = rec_s + rec_t
            
            # domain loss
            transfer_loss1 = Adversarial_loss(s_domain_output1, t_domain_output1) 
            coral_loss_ccc = lam1 * coral(s_ccc_embed, t_ccc_embed)
            domain_loss = transfer_loss1 + 10*coral_loss_ccc
            
            loss = domain_loss + recon_loss  
            total_loss += loss.cpu().detach().item() 
            
    return total_loss

def train_step(writer,epoch,alpha,lam1,cancer_encoder, ccc_encoder,ccc_gate_layer,decoder_cancer, discriminator1,s_batch, t_batch, device, optimizer):
    s_gene_all = s_batch[0].to(device)    # torch.Size([128, 1306])
    s_ccc = s_batch[1].to(device) 
    t_gene_all = t_batch[0].to(device)
    t_ccc = t_batch[1].to(device)

    s_cancer = cancer_encoder(s_gene_all)                 # torch.Size([128, 64])
    s_ccc_embed = ccc_encoder(s_ccc)         # torch.Size([128, 64])?
    s_ccc_gate = torch.sigmoid(ccc_gate_layer(s_ccc_embed))
    s_contextual_gene = s_cancer * s_ccc_gate              
    s_fea_rec1 = decoder_cancer(s_contextual_gene)     
    s_reverse_feature = ReverseLayerF.apply(s_contextual_gene, alpha) 
    s_domain_output1 = discriminator1(s_reverse_feature)    # torch.Size([128, 1])
    

    t_cancer = cancer_encoder(t_gene_all)
    t_ccc_embed = ccc_encoder(t_ccc)
    t_ccc_gate = torch.sigmoid(ccc_gate_layer(t_ccc_embed))
    t_contextual_gene = t_cancer * t_ccc_gate
    t_fea_rec1 = decoder_cancer(t_contextual_gene)
    t_reverse_feature = ReverseLayerF.apply(t_contextual_gene, alpha)  
    t_domain_output1 = discriminator1(t_reverse_feature)
    

    # reconstruct loss
    rec_s = F.mse_loss(s_gene_all, s_fea_rec1)
    rec_t = F.mse_loss(t_gene_all, t_fea_rec1)
    recon_loss = rec_s + rec_t

    # domain loss
    transfer_loss1 = Adversarial_loss(s_domain_output1, t_domain_output1)  
    
    coral_loss_ccc = lam1 * coral(s_ccc_embed, t_ccc_embed)
    domain_loss = transfer_loss1 +  10*coral_loss_ccc

    loss = domain_loss + recon_loss  
    loss.backward()
    optimizer.step()

    
    return loss.cpu().detach().item() 

def _build_ccc_encoder(**kwargs):
    ccc_encoder_type = kwargs.get("ccc_encoder_type", "bipartite")
    if ccc_encoder_type == "mlp":
        return extractionForCCC(
            dim_input=kwargs["ccc_num"],
            dim_hid=kwargs["cancer_encoder_hidden_dims"][1],
            dim_final=kwargs["cancer_encoder_hidden_dims"][2],
        ).to(kwargs["device"])
    if ccc_encoder_type == "bipartite":
        return BipartiteCCC(
            ligand_gene_ids=kwargs["ligand_gene_ids"],
            receptor_gene_ids=kwargs["receptor_gene_ids"],
        ).to(kwargs["device"])
    raise ValueError(f"Unsupported ccc_encoder_type: {ccc_encoder_type}")


def training(base_edge_index,s_dataloaders, t_dataloaders, **kwargs):
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]
    
    cancer_encoder = extractionForCancer(dim_input=kwargs['cancer_genes'], 
                                         dim_hid1=kwargs['cancer_encoder_hidden_dims'][0],
                                         dim_hid2=kwargs['cancer_encoder_hidden_dims'][1], 
                                         dim_final=kwargs['cancer_encoder_hidden_dims'][2]).to(kwargs['device'])   

    ccc_encoder = _build_ccc_encoder(**kwargs)


    discriminator1 = Discriminator(input_dim=kwargs['cancer_encoder_hidden_dims'][2]).to(kwargs['device'])  
    ccc_gate_layer = torch.nn.Linear(kwargs['latent_dim'], kwargs['latent_dim']).to(kwargs['device'])
    
    decoder_cancer = Decoder(input_dim = kwargs['cancer_encoder_hidden_dims'][2],
                      output_dim = kwargs['cancer_genes'],
                      hidden_dims = kwargs['decoder_hidden_dims'],
                      drop = kwargs['drop']).to(kwargs['device'])
    
    params = [cancer_encoder.parameters(),
              ccc_encoder.parameters(),
              decoder_cancer.parameters(),
              ccc_gate_layer.parameters(),
              discriminator1.parameters()]
    
    optimizer = torch.optim.AdamW(chain(*params), lr=kwargs['lr'])
    best_threshold = np.inf


    print("================Pre-training for feature extraction================")
    if kwargs['retrain_flag']:
        # starting pre-training for domain feature  
        writer = SummaryWriter(f"logs/stage1_{kwargs.get('ccc_encoder_type', 'bipartite')}")
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            train_loss_all = 0
            val_loss_all = 0
            len_loader = min(len(s_train), len(t_train))   
            cancer_encoder.train()
            ccc_encoder.train()
            ccc_gate_layer.train()
            decoder_cancer.train()
            discriminator1.train()
            
            optimizer.zero_grad()
            for i, batch in enumerate(zip(t_train, cycle(s_train))):
                p = float(i + epoch * len_loader) / int(kwargs['pretrain_num_epochs']) / len_loader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                train_loss = train_step(writer,epoch,alpha,kwargs['lam1'],cancer_encoder, ccc_encoder,ccc_gate_layer,
                                        decoder_cancer, discriminator1, 
                                        s_batch=batch[1], t_batch=batch[0],
                                        device=kwargs['device'], optimizer=optimizer)
                train_loss_all += train_loss
            print('Pre-training epoch = {}, training loss = {:.4f}'.format(epoch+1, train_loss_all))

            cancer_encoder.eval()
            ccc_encoder.eval()
            ccc_gate_layer.eval()
            decoder_cancer.eval()
            discriminator1.eval()

            val_loss = eval_epoch(kwargs['lam1'],cancer_encoder, ccc_encoder,ccc_gate_layer,decoder_cancer, discriminator1,loader1=s_test, loader2=t_test,device=kwargs['device'])
            val_loss_all = val_loss
            print('validate loss = {:.4f}'.format(val_loss_all))

            if (val_loss_all < best_threshold):
                torch.save(cancer_encoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt'))
                torch.save(ccc_encoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'ccc_encoder.pt'))   
                torch.save(ccc_gate_layer.state_dict(), os.path.join(kwargs['model_save_folder'], 'ccc_gate_layer.pt')) 
                best_threshold = val_loss_all
            if(epoch+1) % 50 == 0:  
                torch.save(cancer_encoder.state_dict(), os.path.join(kwargs['model_save_folder'],'epoch_{}'.format(epoch+1)+'_cancer_encoder.pt'))
                torch.save(ccc_encoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'epoch_{}'.format(epoch+1)+'_ccc_encoder.pt'))  
                torch.save(ccc_gate_layer.state_dict(), os.path.join(kwargs['model_save_folder'], 'epoch_{}'.format(epoch+1)+'_ccc_gate_layer.pt')) 
        writer.close()
        cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt'), map_location=kwargs['device']))
        ccc_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ccc_encoder.pt'), map_location=kwargs['device']))
        ccc_gate_layer.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ccc_gate_layer.pt'), map_location=kwargs['device']))
    else:
        try:
            cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt'), map_location=kwargs['device']))
            ccc_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ccc_encoder.pt'), map_location=kwargs['device']))
            ccc_gate_layer.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ccc_gate_layer.pt'), map_location=kwargs['device']))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    # t_AE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_AE.pt')))    
    
    return cancer_encoder, ccc_encoder,ccc_gate_layer


def training2(base_edge_index, s_dataloaders, t_dataloaders, **kwargs):
    if kwargs['retrain_flag']:
        return training(base_edge_index, s_dataloaders, t_dataloaders, **kwargs)

    cancer_encoder = extractionForCancer(dim_input=kwargs['cancer_genes'],
                                         dim_hid1=kwargs['cancer_encoder_hidden_dims'][0],
                                         dim_hid2=kwargs['cancer_encoder_hidden_dims'][1],
                                         dim_final=kwargs['cancer_encoder_hidden_dims'][2]).to(kwargs['device'])
    ccc_encoder = _build_ccc_encoder(**kwargs)
    ccc_gate_layer = torch.nn.Linear(kwargs['latent_dim'], kwargs['latent_dim']).to(kwargs['device'])

    pretrained_folder = kwargs.get('pretrained_model_folder', kwargs['model_save_folder'])
    try:
        cancer_encoder.load_state_dict(torch.load(os.path.join(pretrained_folder, 'cancer_encoder.pt'), map_location=kwargs['device']))
        ccc_encoder.load_state_dict(torch.load(os.path.join(pretrained_folder, 'ccc_encoder.pt'), map_location=kwargs['device']))
        ccc_gate_layer.load_state_dict(torch.load(os.path.join(pretrained_folder, 'ccc_gate_layer.pt'), map_location=kwargs['device']))
    except FileNotFoundError:
        raise Exception(f"No pre-trained encoder in {pretrained_folder}")

    return cancer_encoder, ccc_encoder, ccc_gate_layer



def eval_epoch1(lam1,cancer_encoder, ccc_encoder,ccc_gate_layer,decoder_cancer, discriminator1,loader1, loader2, device):
    total_loss = 0

    for i, batch in enumerate(zip(loader2, cycle(loader1))):
        s_batch=batch[1]
        t_batch=batch[0]
        s_gene_all = s_batch[0].to(device)    # torch.Size([128, 1306])
        s_ccc = s_batch[1].to(device) # torch.Size([128, 732])
        t_gene_all = t_batch[0].to(device)
        t_ccc = t_batch[1].to(device)
                

        with torch.no_grad():
            s_cancer = cancer_encoder(s_gene_all)                 # torch.Size([128, 64])
            # s_ccc_embed = ccc_encoder(s_ccc)
            # s_ccc_gate = torch.sigmoid(ccc_gate_layer(s_ccc_embed))
            s_contextual_gene = s_cancer# * s_ccc_gate
            s_domain_output1 = discriminator1(s_contextual_gene)             # torch.Size([128, 1])
            s_fea_rec1 = decoder_cancer(s_contextual_gene)                        # torch.Size([128, ??])

            t_cancer = cancer_encoder(t_gene_all)
            # t_ccc_embed = ccc_encoder(t_ccc)
            # t_ccc_gate = torch.sigmoid(ccc_gate_layer(t_ccc_embed))
            t_contextual_gene = t_cancer# * t_ccc_gate
            t_domain_output1 = discriminator1(t_contextual_gene)
            t_fea_rec1 = decoder_cancer(t_contextual_gene)

    
            # reconstruct loss
            rec_s = F.mse_loss(s_gene_all, s_fea_rec1)
            rec_t = F.mse_loss(t_gene_all, t_fea_rec1)
            recon_loss = rec_s + rec_t
            
            # domain loss
            transfer_loss1 = Adversarial_loss(s_domain_output1, t_domain_output1)  
            # coral_loss_ccc = lam1 * coral(s_ccc_embed, t_ccc_embed)
            domain_loss = transfer_loss1 #+ 10*coral_loss_ccc
            
            loss = recon_loss + domain_loss
            total_loss += loss.cpu().detach().item() 
            
    return total_loss

def train_step1(writer,epoch,alpha,lam1,cancer_encoder, ccc_encoder,ccc_gate_layer,decoder_cancer, discriminator1,s_batch, t_batch, device, optimizer):
    s_gene_all = s_batch[0].to(device)    # torch.Size([128, 1306])
    s_ccc = s_batch[1].to(device) 
    t_gene_all = t_batch[0].to(device)
    t_ccc = t_batch[1].to(device)

    s_cancer = cancer_encoder(s_gene_all)                 # torch.Size([128, 64])
    # s_ccc_embed = ccc_encoder(s_ccc)         # torch.Size([128, 64])?
    # s_ccc_gate = torch.sigmoid(ccc_gate_layer(s_ccc_embed))
    s_contextual_gene = s_cancer #* s_ccc_gate             
    s_fea_rec1 = decoder_cancer(s_contextual_gene)     
    s_reverse_feature = ReverseLayerF.apply(s_contextual_gene, alpha)  
    s_domain_output1 = discriminator1(s_reverse_feature)    # torch.Size([128, 1])
    

    t_cancer = cancer_encoder(t_gene_all)
    # t_ccc_embed = ccc_encoder(t_ccc)
    # t_ccc_gate = torch.sigmoid(ccc_gate_layer(t_ccc_embed))
    t_contextual_gene = t_cancer # * t_ccc_gate
    t_fea_rec1 = decoder_cancer(t_contextual_gene)
    t_reverse_feature = ReverseLayerF.apply(t_contextual_gene, alpha)   
    t_domain_output1 = discriminator1(t_reverse_feature)
    

    # reconstruct loss
    rec_s = F.mse_loss(s_gene_all, s_fea_rec1)
    rec_t = F.mse_loss(t_gene_all, t_fea_rec1)
    recon_loss = rec_s + rec_t

    # domain loss
    transfer_loss1 = Adversarial_loss(s_domain_output1, t_domain_output1)  
     
    domain_loss = transfer_loss1 #+ 10*coral_loss_ccc
  

    loss = recon_loss + domain_loss
    loss.backward()
    optimizer.step()


    
    
    return loss.cpu().detach().item() 

def training1(base_edge_index,s_dataloaders, t_dataloaders, **kwargs):
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]
    
    cancer_encoder = extractionForCancer(dim_input=kwargs['cancer_genes'], 
                                         dim_hid1=kwargs['cancer_encoder_hidden_dims'][0],
                                         dim_hid2=kwargs['cancer_encoder_hidden_dims'][1], 
                                         dim_final=kwargs['cancer_encoder_hidden_dims'][2]).to(kwargs['device'])   

    ccc_encoder = None
  
    discriminator1 = Discriminator(input_dim=kwargs['cancer_encoder_hidden_dims'][2]).to(kwargs['device'])
  
    ccc_gate_layer = None
    
    decoder_cancer = Decoder(input_dim = kwargs['cancer_encoder_hidden_dims'][2],
                      output_dim = kwargs['cancer_genes'],
                      hidden_dims = kwargs['decoder_hidden_dims'],
                      drop = kwargs['drop']).to(kwargs['device'])
    
    params = [cancer_encoder.parameters(),
              decoder_cancer.parameters(),
              discriminator1.parameters()]
    
    optimizer = torch.optim.AdamW(chain(*params), lr=kwargs['lr'])
    best_threshold = np.inf


    print("================Pre-training for feature extraction================")
    if kwargs['retrain_flag']:
        writer = SummaryWriter("logs/stage1")    
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            train_loss_all = 0
            val_loss_all = 0
            len_loader = min(len(s_train), len(t_train))    
            cancer_encoder.train()
            decoder_cancer.train()
            discriminator1.train()
            
            optimizer.zero_grad()
            for i, batch in enumerate(zip(t_train, cycle(s_train))):
                p = float(i + epoch * len_loader) / int(kwargs['pretrain_num_epochs']) / len_loader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                train_loss = train_step1(writer,epoch,alpha,kwargs['lam1'],cancer_encoder, ccc_encoder,ccc_gate_layer,
                                        decoder_cancer, discriminator1, 
                                        s_batch=batch[1], t_batch=batch[0],
                                        device=kwargs['device'], optimizer=optimizer)
                train_loss_all += train_loss
            print('Pre-training epoch = {}, training loss = {:.4f}'.format(epoch+1, train_loss_all))



            cancer_encoder.eval()
            decoder_cancer.eval()
            discriminator1.eval()

            val_loss = eval_epoch1(kwargs['lam1'],cancer_encoder, ccc_encoder,ccc_gate_layer,decoder_cancer, discriminator1,loader1=s_test, loader2=t_test,device=kwargs['device'])
            val_loss_all = val_loss
            print('validate loss = {:.4f}'.format(val_loss_all))

            if (val_loss_all < best_threshold):
                torch.save(cancer_encoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt'))  
                best_threshold = val_loss_all
            if(epoch+1) % 50 == 0: 
                torch.save(cancer_encoder.state_dict(), os.path.join(kwargs['model_save_folder'],'epoch_{}'.format(epoch+1)+'_cancer_encoder.pt'))
        writer.close()
        cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt'), map_location=kwargs['device']))
    else:
        try:
            cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt'), map_location=kwargs['device']))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    
    return cancer_encoder#, ccc_encoder,ccc_gate_layer
