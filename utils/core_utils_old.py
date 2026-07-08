import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import torch.nn.functional as F
from models.infonce import InfoNCE
import wandb
from thop import profile


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def _safe_add_scalar(writer, tag, value, step):
    import numpy as np, torch, math
    if value is None:
        print(f"[TBX] Skip {tag}: value=None (no samples)")
        return
    if torch.is_tensor(value):
        if value.numel() == 0:
            print(f"[TBX] Skip {tag}: empty tensor")
            return
        value = value.detach().float().mean().item()
    else:
        try:
            value = float(np.asarray(value).mean())
        except Exception as e:
            print(f"[TBX] Skip {tag}: cannot convert ({e})")
            return
    if math.isnan(value):
        print(f"[TBX] Skip {tag}: NaN")
        return
    writer.add_scalar(tag, value, step)

def measure_expected_flops(model, loader, args, device, is_cgrl=True):
    """
    Hàm đo lường Expected FLOPs theo 2 bước:
    1. Quét tập dữ liệu để lấy N_avg (số patch ban đầu) và K_avg (số patch giữ lại).
    2. Dùng N_avg và K_avg để tính FLOPs thực tế.
    """
    model.eval() # Đảm bảo mô hình ở chế độ test
    total_n = 0
    total_k = 0
    num_bags = len(loader)

    if num_bags == 0:
        return

    print('\n[FLOPs Evaluation] Đang duyệt qua tập Dữ liệu để thống kê N_avg và K_avg...')
    with torch.inference_mode():
        for data, _ in loader:
            data = data.to(device)
            n_patches = data.size(1) if data.dim() == 3 else data.size(0)
            total_n += n_patches

            if is_cgrl:
                # Truyền qua selector để xem mô hình ĐÃ TRAIN sẽ giữ lại chính xác bao nhiêu patch
                x_selected, _ = model.selector(data, label=None)
                k_patches = x_selected.size(1) if x_selected.dim() == 3 else x_selected.size(0)
                total_k += k_patches
            else:
                total_k += n_patches

    n_avg = max(1, int(total_n / num_bags))
    k_avg = max(1, int(total_k / num_bags))
    keep_ratio = (k_avg / n_avg) * 100

    print(f"--> Thống kê xong: N_avg = {n_avg} patches, K_avg = {k_avg} patches (Giữ lại {keep_ratio:.2f}%)")

    print('[FLOPs Evaluation] Đang tính toán Expected FLOPs...')
    try:
        from thop import profile
        # Tạo dummy input dạng [1, N, D]
        dummy_N = torch.randn(1, n_avg, args.embed_dim).to(device)
        dummy_K = torch.randn(1, k_avg, args.embed_dim).to(device)

        if is_cgrl:
            # 1. Baseline MIL (Mạng gốc nếu không có CGRL thì phải cõng toàn bộ N_avg)
            flops_base, params_base = profile(model.mil, inputs=(dummy_N, ), verbose=False)

            # 2. CGRL (Selector cõng N_avg + MIL cõng K_avg)
            flops_sel, params_sel = profile(model.selector, inputs=(dummy_N, ), verbose=False)
            flops_mil_k, _ = profile(model.mil, inputs=(dummy_K, ), verbose=False)

            total_flops_cgrl = flops_sel + flops_mil_k
            total_params_cgrl = params_sel + params_base

            print(f"\n" + "="*60)
            print(f" BÁO CÁO EXPECTED FLOPs TRÊN TẬP DỮ LIỆU THỰC TẾ")
            print("="*60)
            print(f"[BASELINE MIL] FLOPs: {flops_base / 1e9:.4f} G | Params: {params_base / 1e6:.4f} M")
            print(f"[CGRL MIL]     FLOPs: {total_flops_cgrl / 1e9:.4f} G | Params: {total_params_cgrl / 1e6:.4f} M")
            
            saved_flops = (1 - total_flops_cgrl / flops_base) * 100 if flops_base > 0 else 0
            print(f"--> CGRL giúp tiết kiệm được: {saved_flops:.2f}% FLOPs")
            print("="*60 + "\n")

            wandb.log({
                "Eval/N_avg": n_avg,
                "Eval/K_avg": k_avg,
                "Eval/Keep_Ratio_%": keep_ratio,
                "Eval/GFLOPs_Baseline": flops_base / 1e9,
                "Eval/GFLOPs_CGRL": total_flops_cgrl / 1e9
            })

        else:
            flops, params = profile(model, inputs=(dummy_N, ), verbose=False)
            print(f"\n[BASELINE MIL] FLOPs: {flops / 1e9:.4f} G | Params: {params / 1e6:.4f} M\n")

    except Exception as e:
        print(f"Lỗi khi tính toán FLOPs: {e}")

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    # ---------------------------------------------------------
    # [CẬP NHẬT MỚI] - KỸ THUẬT PLUG AND PLAY
    # ---------------------------------------------------------
    # Kiểm tra xem user có muốn dùng bộ lọc CGRL (ConceptSelector) không
    is_cgrl = args.model_type.startswith('cgrl_')
    base_model_type = args.model_type.replace('cgrl_', '') if is_cgrl else args.model_type

    if base_model_type in ['clam_sb', 'clam_mb','cate_abmil','cate_dl','abmil','mean_mil','max_mil','trans_mil','mamba_mil','dsmil']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
            
        if args.inst_loss == 'svm' and base_model_type in ['clam_sb', 'clam_mb']:
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        model_dictionary = {
                "input_dim": args.embed_dim,
                "n_classes": args.n_classes,
                "dropout": args.drop_out}

        # KHỞI TẠO MẠNG MIL LÕI
        if base_model_type == 'mean_mil':
            from models.Mean_Max_MIL import MeanMIL
            model = MeanMIL(args.embed_dim, args.n_classes)
        elif base_model_type == 'dsmil':
            from models.dsmil import FCLayer , IClassifier, BClassifier,MILNet
            i_classifier = FCLayer(in_size=args.embed_dim, out_size=args.n_classes)
            b_classifier = BClassifier(input_size=args.embed_dim, output_class=args.n_classes, dropout_v=0.0)
            model = MILNet(i_classifier, b_classifier)
        elif base_model_type == 'max_mil':
            from models.Mean_Max_MIL import MaxMIL
            model = MaxMIL(args.embed_dim, args.n_classes)
        elif base_model_type == 'trans_mil':
            from models.TransMIL import TransMIL
            model = TransMIL(args.embed_dim, args.n_classes, dropout=args.drop_out, act='relu')
        elif base_model_type == 'mamba_mil':
            from models.MambaMIL import MambaMIL
            model = MambaMIL(in_dim = args.embed_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)
        elif base_model_type =='clam_sb':
            from models.model_clam import CLAM_SB
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif base_model_type == 'clam_mb':
            from models.model_clam import CLAM_MB
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif base_model_type == 'abmil':
            from models.abmil import DAttention
            model = DAttention(**model_dictionary)    
        else:
            raise NotImplementedError

        # # BỌC MẠNG VỪA KHỞI TẠO VÀO MODULE CHỌN LỌC (WRAPPER)
        # if is_cgrl:
        #     from models.cgrl_mil import CGRLMIL # Đảm bảo file import đúng đường dẫn của bạn
        #     num_selected = getattr(args, 'num_selected', 100) # Mặc định lấy 100 vector nếu không có arg
        #     model = CGRLMIL(
        #         base_mil_model=model, 
        #         input_dim=args.embed_dim, 
        #         n_classes=args.n_classes, 
        #         concept_path=args.concept_path, 
        #         num_selected=num_selected
        #     )
        # BỌC MẠNG VỪA KHỞI TẠO VÀO MODULE CHỌN LỌC (WRAPPER)
        if is_cgrl:
            from models.cgrl_mil import CGRLMIL # Đảm bảo file import đúng đường dẫn của bạn
            model = CGRLMIL(
                base_mil_model=model, 
                input_dim=args.embed_dim, 
                compress_dim=args.compress_dim, # Truyền tham số chiều nén vào đây
                n_classes=args.n_classes, 
                concept_path=args.concept_path
            )
    # ---------------------------------------------------------

    _ = model.to(device)
    print('Done!')
    print_network(model)

    

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')


    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
                
        # [MỚI] Điều hướng vào vòng lặp của CGRL
        elif is_cgrl:
            train_loop_cgrl(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,args)
            stop = validate_cgrl(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
                
        # [CŨ] Vòng lặp mặc định cho các mô hình còn lại
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # =========================================================
    # GỌI HÀM TÍNH FLOPs TỔNG QUÁT TẠI ĐÂY (SAU KHI ĐÃ LOAD MODEL TỐT NHẤT)
    # =========================================================
    print("\n--- BẮT ĐẦU ĐÁNH GIÁ ĐỘ PHỨC TẠP MÔ HÌNH (TEST SET) ---")
    is_cgrl = args.model_type.startswith('cgrl_')
    # Chạy hàm đo lường trên tập Test để đánh giá khách quan
    measure_expected_flops(model, test_loader, args, device, is_cgrl=is_cgrl)
    # =========================================================

    # === ĐIỀU HƯỚNG BƯỚC TEST TỔNG KẾT ===
    if is_cgrl:
        _, val_error, val_auc, _ = summary_cgrl(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
        results_dict, test_error, test_auc, acc_logger = summary_cgrl(model, test_loader, args.n_classes)
    else:
        _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)

    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    # --- [THÊM MỚI] ĐÓNG GÓI DỮ LIỆU TEST TỔNG KẾT CHO WANDB ---
    wandb_final_logs = {
        'final/val_error': val_error,
        'final/val_auc': val_auc,
        'final/test_error': test_error,
        'final/test_auc': test_auc
    }

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            _safe_add_scalar(writer, f'final/test_class_{i}_acc', acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    # Đẩy bảng tổng kết lên WandB
    wandb.log(wandb_final_logs)
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


# ===================================================================================
# CÁC HÀM DÀNH RIÊNG CHO MÔ HÌNH CGRL (PLUG AND PLAY)
# ===================================================================================

# def train_loop_cgrl(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):   
#     model.train()
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     train_loss = 0.
#     train_error = 0.
#     train_sim_loss = 0.

#     print('\n')
#     for batch_idx, (data, label) in enumerate(loader):
#         data, label = data.to(device), label.to(device)

#         # Mô hình CGRL cần nhận label để tính sim_loss và trả về results_dict
#         logits, Y_prob, Y_hat, results_dict, _ = model(data, label=label)
        
#         acc_logger.log(Y_hat, label)
        
#         # Loss phân loại túi (CrossEntropy, SVM...)
#         cls_loss = loss_fn(logits, label)
        
#         # Lấy Sim Loss (nếu có)
#         sim_loss = results_dict.get('sim_loss', torch.tensor(0.0, device=device))
        
#         # Tổng hợp loss
#         loss = cls_loss + 0.01*sim_loss
#         loss_value = loss.item()
        
#         train_loss += loss_value
#         train_sim_loss += sim_loss.item() if torch.is_tensor(sim_loss) else sim_loss

#         if (batch_idx + 1) % 20 == 0:
#             print('batch {}, total_loss: {:.4f}, cls_loss: {:.4f}, sim_loss: {:.4f}, label: {}, bag_size: {}'.format(
#                 batch_idx, loss_value, cls_loss.item(), 
#                 sim_loss.item() if torch.is_tensor(sim_loss) else sim_loss, 
#                 label.item(), data.size(0)))
           
#         error = calculate_error(Y_hat, label)
#         train_error += error
        
#         # Backward & Update
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#     # Calculate epoch averages
#     train_loss /= len(loader)
#     train_error /= len(loader)
#     train_sim_loss /= len(loader)

#     print('Epoch: {}, train_loss: {:.4f}, sim_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_sim_loss, train_error))
#     # --- [THÊM MỚI] ĐÓNG GÓI DỮ LIỆU ĐỂ GỬI LÊN WANDB ---
#     wandb_train_logs = {
#         'train/loss': train_loss,
#         'train/sim_loss': train_sim_loss,
#         'train/error': train_error,
#         'epoch': epoch
#     }
#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
#         if writer:
#             writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

#     if writer:
#         writer.add_scalar('train/loss', train_loss, epoch)
#         writer.add_scalar('train/sim_loss', train_sim_loss, epoch)
#         writer.add_scalar('train/error', train_error, epoch)
    
#     wandb.log(wandb_train_logs)

def train_loop_cgrl(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, args=None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_sim_loss = 0.
    train_cac_loss = 0. # Thêm biến theo dõi cac_loss

    # Lấy hệ số từ args (nếu không truyền vào, mặc định là 0.01 và 0.1)
    alpha = args.alpha if args and hasattr(args, 'alpha') else 0.01
    beta = args.beta if args and hasattr(args, 'beta') else 0.1

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, results_dict, _ = model(data, label=label)
        
        acc_logger.log(Y_hat, label)
        
        cls_loss = loss_fn(logits, label)
        sim_loss = results_dict.get('sim_loss', torch.tensor(0.0, device=device))
        cac_loss = results_dict.get('cac_loss', torch.tensor(0.0, device=device))
        
        # --- ÁP DỤNG TRỌNG SỐ LOSS Ở ĐÂY ---
        loss = cls_loss + (alpha * sim_loss) + (beta * cac_loss)
        loss_value = loss.item()
        
        train_loss += loss_value
        train_sim_loss += sim_loss.item() if torch.is_tensor(sim_loss) else sim_loss
        train_cac_loss += cac_loss.item() if torch.is_tensor(cac_loss) else cac_loss

        if (batch_idx + 1) % 20 == 0:
            print('batch {}, total_loss: {:.4f}, cls_loss: {:.4f}, sim: {:.4f}, cac: {:.4f}, label: {}, bag: {}'.format(
                batch_idx, loss_value, cls_loss.item(), 
                sim_loss.item() if torch.is_tensor(sim_loss) else sim_loss, 
                cac_loss.item() if torch.is_tensor(cac_loss) else cac_loss,
                label.item(), data.size(0)))
            
        error = calculate_error(Y_hat, label)
        train_error += error
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)
    train_sim_loss /= len(loader)
    train_cac_loss /= len(loader)

    print('Epoch: {}, total_loss: {:.4f}, sim_loss: {:.4f}, cac_loss: {:.4f}, train_error: {:.4f}'.format(
        epoch, train_loss, train_sim_loss, train_cac_loss, train_error))
    
    # --- ĐÓNG GÓI DỮ LIỆU ĐỂ GỬI LÊN WANDB ---
    wandb_train_logs = {
        'train/loss': train_loss,
        'train/sim_loss': train_sim_loss,
        'train/cac_loss': train_cac_loss,
        'train/error': train_error,
        'epoch': epoch
    }
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/sim_loss', train_sim_loss, epoch)
        writer.add_scalar('train/cac_loss', train_cac_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
    
    wandb.log(wandb_train_logs)

def validate_cgrl(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            # Đặt label=None để bỏ qua hoàn toàn việc tính toán sim_loss bên trong model (tăng tốc độ)
            logits, Y_prob, Y_hat, _, _ = model(data, label=None)

            acc_logger.log(Y_hat, label)
            
            # Chỉ dùng classification loss
            cls_loss = loss_fn(logits, label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            # EarlyStopping giờ đánh giá hoàn toàn dựa trên val_loss thuần túy
            val_loss += cls_loss.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error
            
    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    # Đóng gói dữ liệu gửi lên WandB (đã xóa sim_loss)
    wandb_val_logs = {
        'val/loss': val_loss,
        'val/error': val_error,
        'val/auc': auc,
        'epoch': epoch
    }

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))  

    # Đẩy dữ liệu lên WandB
    wandb.log(wandb_val_logs)   

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True
            
    return False


def summary_cgrl(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            # Trong lúc test, gán nhãn label=None để tắt sim_loss, tăng tốc mô hình
            logits, Y_prob, Y_hat, _, _ = model(data, label=None)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger