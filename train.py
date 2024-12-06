import os, argparse, time
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy.io as scio
from scipy import stats
from scipy.optimize import curve_fit
from AFQNet import AFQNet
from utils.MyDataset import MyDataset
from utils.loss import CTFLoss


def set_rand_seed(seed=3407):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat
    
def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic


def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=8, type=int)
    parser.add_argument('--test_patch_num', help='Test patch number.', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.00002 , type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=5e-4, help='decay rate')
    parser.add_argument('--data_dir_texture', default='/home/old/zhangyujie/database/WPC/proj_6view_1angle_512', type=str, help = 'path to the images')
    parser.add_argument('--data_dir_depth', default='/home/old/zhangyujie/database/WPC/proj_6view_1angle_512_depth', type=str, help = 'path to the depth maps')
    parser.add_argument('--data_dir_mask', default='/home/old/zhangyujie/database/WPC/proj_6view_1angle_512_mask', type=str, help = 'path to the mask maps')
    parser.add_argument('--output_dir', default='./results/', type=str, help = 'path to the saved models')
    parser.add_argument('--save_flag', help="Flag of saving trained models", default=True, type=bool)
    parser.add_argument('--database', default='WPC', type=str)
    parser.add_argument('--k_fold_num', default=5, type=int, help='9 for SJTU-PCQA, 5 for LS-PCQA, WPC, and BASICS, 4 for M-PCCD')
    args = parser.parse_args()
    return args

def main(args):
    print('*************************************************************************************************************************')

    cudnn.enabled = True
    save_flag = args.save_flag

    output_dir = args.output_dir
    if save_flag:
        os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    database = args.database
    test_patch_num = args.test_patch_num

    data_dir_texture = args.data_dir_texture
    data_dir_depth = args.data_dir_depth
    data_dir_mask = args.data_dir_mask
  
    best_all = np.zeros([args.k_fold_num, 4])

    for k_fold_id in range(1,args.k_fold_num + 1):
        
        print('The current k_fold_id is ' + str(k_fold_id))
        if database == 'SJTU':           
            train_filename_list = 'csvfiles/sjtu_data_info/train_'+str(k_fold_id)+'.csv'
            test_filename_list = 'csvfiles/sjtu_data_info/test_'+str(k_fold_id)+'.csv'
        elif database == 'WPC':
            train_filename_list = 'csvfiles/wpc_data_info/train_'+str(k_fold_id)+'.csv'
            test_filename_list = 'csvfiles/wpc_data_info/test_'+str(k_fold_id)+'.csv'
        elif database == 'LSPCQA':
            train_filename_list = 'csvfiles/ls_pcqa_data_info/train_'+str(k_fold_id)+'.csv'
            test_filename_list = 'csvfiles/ls_pcqa_data_info/test_'+str(k_fold_id)+'.csv'
        elif database == 'MPCCD':
            train_filename_list = 'csvfiles/mpccd_data_info/train_'+str(k_fold_id)+'.csv'
            test_filename_list = 'csvfiles/mpccd_data_info/test_'+str(k_fold_id)+'.csv'
        elif database == 'BASICS':
            train_filename_list = 'csvfiles/basics_data_info/train_'+str(k_fold_id)+'.csv'
            test_filename_list = 'csvfiles/basics_data_info/test_'+str(k_fold_id)+'.csv'
    
        transformations_train = transforms.Compose([transforms.RandomCrop(224),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406, 0.5, 0], std = [0.229, 0.224, 0.225, 0.5,1])])
       
        transformations_test = transforms.Compose([transforms.RandomCrop(224),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406, 0.5, 0], std = [0.229, 0.224, 0.225, 0.5,1 ])])
        
        print('Trainging set: ' + train_filename_list)
        
        model = AFQNet()
        model = model.to(device)

        criterion = CTFLoss().to(device)
        print('Using corase-to-fine loss')

        encoder_params = list(map(id, model.ViT.encoder.parameters()))  
        other_params = filter(lambda p: id(p) not in encoder_params, model.ViT.parameters())
        
        paras = [{'params': model.ViT.encoder.parameters(), 'lr': args.learning_rate},
                 {'params': other_params, 'lr': args.learning_rate*10},
                 {'params': model.HyperNet.parameters(), 'lr': args.learning_rate*10}
                 ]
        
        optimizer = torch.optim.Adam(paras, weight_decay=args.decay_rate)
        print('Using Adam optimizer, initial learning rate: ' + str(args.learning_rate))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        print("Ready to train network")
        print('*************************************************************************************************************************')

        best = np.zeros(4)
        min_training_loss = 10000
        
        train_dataset = MyDataset(data_dir_texture = data_dir_texture, data_dir_depth = data_dir_depth, data_dir_mask = data_dir_mask, datainfo_path = train_filename_list, transform = transformations_train, patch_num = 1)
        test_dataset = MyDataset(data_dir_texture = data_dir_texture, data_dir_depth = data_dir_depth, data_dir_mask = data_dir_mask,  datainfo_path = test_filename_list, transform = transformations_test, patch_num=test_patch_num, is_train = False)
        
        columns = ['Epoch', 'Train_Loss', 'Train_SRCC', 'Test_SRCC', 'Test_PLCC', 'Training_time(s)']
        results_df = pd.DataFrame(columns=columns)
        
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTraining_time(s)')
        
        for epoch in range(num_epochs):
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last = True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_patch_num , shuffle=False, num_workers=8, drop_last = True)
            n_train = len(train_loader)
            n_test = len(test_loader)

            model.train()
    
            start = time.time()
            batch_losses = []
            x_pre = []
            x_gt = []
            for i, (imgs, mos) in enumerate(train_loader):
                
                # forward process
                imgs = imgs.to(device)
                mos = mos.to(device)
                output = model(imgs)
                loss = criterion(output, mos)

                batch_losses.append(loss.item())
                x_pre = x_pre + output['score_fine'].tolist()
                x_gt = x_gt + mos.tolist()

                optimizer.zero_grad()   
                torch.autograd.backward(loss)
                optimizer.step()
            x_pre = np.array(x_pre).reshape(-1)
            x_gt = np.array(x_gt).reshape(-1)
            train_SROCC, _ = stats.spearmanr(x_pre, x_gt)
   
            avg_loss = sum(batch_losses) / n_train
            scheduler.step()

            end = time.time()
            train_time = end - start 
            
            # Test 
            model.eval()
    
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)

            with torch.no_grad():
                for i, (imgs, mos) in enumerate(test_loader):
                    imgs = imgs.to(device)
                    mos = torch.mean(mos)
                    y_test[i] = mos.item()
                    output = model(imgs)
                    pre_fine = torch.mean(output['score_fine'])         
                    y_output[i] = pre_fine.item()
                y_output_logistic = fit_function(y_test, y_output)

                test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
                test_SROCC = stats.spearmanr(y_output, y_test)[0]
                test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
                test_KROCC = stats.kendalltau(y_output, y_test)[0]


                print('%-3d\t%-8.3f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f' %
                    (epoch + 1, avg_loss, train_SROCC, test_SROCC, test_PLCC, train_time))
                
                results_df = results_df.append({
                    'Epoch': epoch + 1,
                    'Train_Loss': avg_loss,
                    'Train_SRCC': train_SROCC,
                    'Test_SRCC': test_SROCC,
                    'Test_PLCC': test_PLCC,
                    'Training_time(s)': train_time
                }, ignore_index=True)

                if avg_loss < min_training_loss:
                    # print("Update best model using best_val_criterion ")
                    if save_flag:
                        output_model_name = os.path.join(output_dir, 'model_' + database + '_fold' + str(k_fold_id) + '.pth')
                        torch.save(model.state_dict(), output_model_name )
                        output_mat_name = os.path.join(output_dir, 'prediction_' + database + '_fold' + str(k_fold_id) + '.mat')
                        scio.savemat(output_mat_name,{'y_pred':y_output,'y_test':y_test})
                    best[0:4] = [test_SROCC, test_PLCC, test_KROCC, test_RMSE]
                    min_training_loss = avg_loss

        if save_flag:
            output_excel_name =  os.path.join(output_dir, 'training_info_' + database + '_' + str(k_fold_id) +'.xlsx')
            results_df.to_excel(output_excel_name, index=False)
            print(f"Training results saved to {output_excel_name}")
        
        best_all[k_fold_id-1, :] = best
        print("The best val results in the fold {}: SROCC={:.4f}, PLCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best[0], best[1], best[2], best[3]))
        print('*************************************************************************************************************************')
    
    # average score
    best_mean = np.mean(best_all, 0)
    print('*************************************************************************************************************************')
    print("The mean val results: SROCC={:.4f}, PLCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean[0], best_mean[1], best_mean[2], best_mean[3]))
    print('*************************************************************************************************************************')
    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    set_rand_seed()
    gpu = args.gpu
    with torch.cuda.device(gpu):
        main(args)