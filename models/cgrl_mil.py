import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo bạn đã import hàm này từ file utils/model của bạn
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# ==============================================================================
# 1. BỘ CHỌN LỌC VECTOR (Lấy cảm hứng từ CICS)
# ==============================================================================
class ConceptSelector(nn.Module):
    """
    Module này đóng vai trò nén và lựa chọn thông tin. 
    Nó sẽ giảm số lượng vector từ N xuống num_selected dựa trên độ tương đồng với Concepts.
    """
    def __init__(self, input_dim=512, concept_dim_hidden=512, n_classes=2, 
                 concept_path=None, device='cuda', num_selected=100):
        super(ConceptSelector, self).__init__()
        self.n_classes = n_classes
        self.device = device
        self.num_selected = num_selected # Số lượng vector muốn giữ lại để đưa vào DAttention
        
        # Đã mở khóa InfoNCE loss
        from models.infonce import InfoNCE
        self.info_nce_loss = InfoNCE(temperature=0.1, reduction='mean', negative_mode='unpaired')
        
        self.concepts = None
        self.sim_projector = None

        if concept_path is not None:
            # Quản lý Random State để không làm trôi dòng của các module khác
            rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            concept_dim = self._load_concepts_and_get_dim(concept_path)
            
            # Mạng chiếu (Projector) để tính similarity
            self.sim_projector = nn.Sequential(
                nn.Linear(input_dim, concept_dim_hidden),      
                nn.ReLU(),
                nn.Linear(concept_dim_hidden, concept_dim)  
            )
            self.sim_projector.apply(initialize_weights) 
            
            torch.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

    def _load_concepts_and_get_dim(self, path):
        try:
            raw_concepts = torch.load(path, map_location=self.device) 
            if isinstance(raw_concepts, list):
                sample_feat = raw_concepts[0]
            else:
                sample_feat = raw_concepts
            concept_dim = sample_feat.shape[-1]

            if isinstance(raw_concepts, list):
                raw_concepts = raw_concepts[:self.n_classes]
            
            processed_concepts = []
            for c in raw_concepts:
                if isinstance(c, torch.Tensor):
                    c = c.to(self.device)
                if c.dim() > 1:
                    c = c.mean(dim=0) 
                processed_concepts.append(c)
            
            self.concepts = torch.stack(processed_concepts, dim=0).detach()
            return concept_dim
        except Exception as e:
            print(f"WARNING: Could not load concepts. Error: {e}")
            self.concepts = None
            return 512

    def forward(self, x, label=None, lambda_sym=1.0, lambda_cross=1.0, topk_pos=20, topk_neg=20):
        # x shape: [N, D]
        results_dict = {'sim_loss': torch.tensor(0.0, device=x.device)}

        if self.sim_projector is None or self.concepts is None:
            # Nếu không có concepts, bỏ qua chọn lọc hoặc cắt bớt mặc định
            k_select = min(self.num_selected, x.size(0))
            return x[:k_select], results_dict

        # 1. Chiếu feature để tính toán similarity
        x_proj = self.sim_projector(x)
        
        # 2. Tính toán độ tương đồng với các Concepts
        X_norm = F.normalize(x_proj, dim=-1)
        T_norm = F.normalize(self.concepts, dim=-1)
        S = X_norm @ T_norm.t() # [N, n_classes]

        # 3. LỰA CHỌN VECTOR (Giảm số lượng vector)
        max_sim_per_instance, _ = S.max(dim=-1) # Lấy giá trị similarity cao nhất của mỗi instance [N]
        
        # Chọn ra top K vector
        k_select = min(self.num_selected, x.size(0))
        _, selected_indices = torch.topk(max_sim_per_instance, k_select, dim=0)
        
        # Sắp xếp lại index để giữ đúng thứ tự không gian ban đầu của WSI
        selected_indices, _ = torch.sort(selected_indices)

        # Trả về feature chưa nén (theo đúng triết lý CICS)
        x_selected = x[selected_indices]

        # 4. Tính toán Sim Loss (chỉ khi có label và đang ở mode training)
        #if self.training and label is not None:
        if label is not None:
            results_dict['sim_loss'] = self._calculate_sim_loss(x_proj, S, label, lambda_sym, lambda_cross, topk_pos, topk_neg)

        return x_selected, results_dict

    def _calculate_sim_loss(self, X, S, label, lambda_sym, lambda_cross, topk_pos, topk_neg):
        T = self.concepts
        N = X.shape[0]
        y = int(label.item())
        
        if N == 0 or T.shape[0] < self.n_classes: 
            return torch.tensor(0.0, device=X.device)

        k_pos = max(1, min(topk_pos, N))
        k_neg = max(1, min(topk_neg, N))

        pos_idx = torch.topk(S, k_pos, dim=0, largest=True).indices
        neg_idx = torch.topk(S, k_neg, dim=0, largest=False).indices

        Ly = torch.tensor(0.0, device=X.device)
        q_y = None 

        # --- TÍNH L_y ---
        if 0 <= y < T.shape[0]:
            idx_pos_y = pos_idx[:, y].unique() 
            idx_neg_y = neg_idx[:, y].unique() 

            if idx_pos_y.numel() > 0 and idx_neg_y.numel() > 0:
                q_y = X[idx_pos_y]      
                neg_y = X[idx_neg_y]    
                t_y = T[y].unsqueeze(0).expand(q_y.size(0), -1) 
                Ly = self.info_nce_loss(q_y, t_y, neg_y)

        # --- TÍNH L_sym ---
        L_sym = torch.tensor(0.0, device=X.device)
        if lambda_sym > 0.0 and self.n_classes > 1:
            Lc_list = []
            for c in range(self.n_classes):
                if c == y: continue
                idx_pos_c = pos_idx[:, c].unique()
                idx_neg_c = neg_idx[:, c].unique()
                
                if idx_pos_c.numel() > 0 and idx_neg_c.numel() > 0:
                    q_c = X[idx_pos_c]
                    neg_c = X[idx_neg_c]
                    t_c = T[c].unsqueeze(0).expand(q_c.size(0), -1)
                    Lc_list.append(self.info_nce_loss(q_c, t_c, neg_c))
            
            if len(Lc_list) > 0:
                L_sym = torch.stack(Lc_list).mean()

        # --- TÍNH L_cross ---
        L_cross = torch.tensor(0.0, device=X.device)
        if lambda_cross > 0.0 and q_y is not None and self.n_classes > 1:
            L_yc_list = []
            for c in range(self.n_classes):
                if c == y: continue
                idx_pos_c = pos_idx[:, c].unique()
                idx_neg_c = neg_idx[:, c].unique()
                
                if idx_pos_c.numel() > 0 and idx_neg_c.numel() > 0:
                    pos_c = X[idx_pos_c]
                    neg_c = X[idx_neg_c] 
                    m_neg_c = neg_c.mean(dim=0, keepdim=True)
                    p_yc = m_neg_c.expand(q_y.size(0), -1)
                    L_yc_list.append(self.info_nce_loss(q_y, p_yc, pos_c))
            
            if len(L_yc_list) > 0:
                L_cross = torch.stack(L_yc_list).mean()

        return Ly + lambda_sym * L_sym + lambda_cross * L_cross
        ##return lambda_sym * L_sym + lambda_cross * L_cross
    


# ==============================================================================
# 3. KẾT HỢP (Wrapper Model Plug-and-Play)
# ==============================================================================
class CGRLMIL(nn.Module):
    """
    Kết nối Selector và BẤT KỲ mạng MIL nào. 
    Dữ liệu sẽ qua bộ chọn lọc để giảm từ N -> K vector trước khi vào mạng tính toán nặng.
    """
    def __init__(self, base_mil_model, input_dim=512, n_classes=2, concept_path=None, num_selected=100):
        super().__init__()
        
        self.selector = ConceptSelector(
            input_dim=input_dim, 
            n_classes=n_classes, 
            concept_path=concept_path, 
            num_selected=num_selected
        )
        
        # Gắn mạng MIL bất kỳ được truyền từ ngoài vào
        self.mil = base_mil_model

    def forward(self, x, label=None, **kwargs):
        # 1. Lọc và lấy subset vector tốt nhất
        x_selected, results_dict = self.selector(x, label)
        
        # 2. Truyền x_selected vào mạng MIL cốt lõi
        # Ghi chú: Đảm bảo mạng MIL của bạn có chuẩn đầu ra là 5 biến này
        x_out, Y_prob, Y_hat, mil_results, feature = self.mil(x_selected)
        
        # 3. Gộp từ điển chứa Loss (vd: sim_loss từ selector + loss nội bộ của MIL nếu có)
        # if mil_results is not None:
        #     results_dict.update(mil_results)
        
        return x_out, Y_prob, Y_hat, results_dict, feature


# ==============================================================================
# WRAPPER: RANDOM SELECTION MIL (Dùng làm Baseline so sánh với CGRL)
# ==============================================================================
class RandomMIL(nn.Module):
    """
    Bọc mạng MIL cốt lõi. 
    Trước khi đưa dữ liệu vào mạng, nó sẽ random chọn đúng K vector.
    """
    def __init__(self, base_mil_model, num_selected=100):
        super().__init__()
        self.mil = base_mil_model
        self.num_selected = num_selected

    def forward(self, x, **kwargs):
        # Kiểm tra xem có bị bọc batch_size = 1 ở đầu không [1, N, D]
        is_batched = False
        if x.dim() == 3 and x.size(0) == 1:
            is_batched = True
            x = x.squeeze(0) # Đưa về [N, D]

        N = x.size(0)
        k_select = min(self.num_selected, N)

        # Sinh các index ngẫu nhiên không trùng lặp
        rand_indices = torch.randperm(N, device=x.device)[:k_select]
        
        # Sắp xếp lại index để giữ đúng thứ tự không gian của WSI (rất quan trọng cho Mamba/TransMIL)
        rand_indices, _ = torch.sort(rand_indices)

        # Lọc ra K patch ngẫu nhiên
        x_selected = x[rand_indices]

        # Trả lại chiều batch nếu cần
        if is_batched:
            x_selected = x_selected.unsqueeze(0)

        # Đưa vào mạng MIL cốt lõi
        return self.mil(x_selected, **kwargs)


# ==============================================================================
# WRAPPER 2: K-MEANS SELECTION MIL (Baseline Phân cụm đặc trưng)
# ==============================================================================
class KMeansMIL(nn.Module):
    """
    Sử dụng K-Means trên GPU để gom cụm N patch thành K cụm.
    Sau đó chọn ra K patch nằm gần tâm của các cụm nhất để đại diện cho toàn bộ WSI.
    """
    def __init__(self, base_mil_model, num_selected=100, max_iters=5):
        super().__init__()
        self.mil = base_mil_model
        self.num_selected = num_selected
        self.max_iters = max_iters # Giới hạn số lần lặp để đảm bảo tốc độ training

    def _get_kmeans_indices(self, x, k):
        N, D = x.shape
        if N <= k:
            return torch.arange(N, device=x.device)

        # 1. Khởi tạo ngẫu nhiên K tâm cụm từ dữ liệu gốc
        indices = torch.randperm(N, device=x.device)[:k]
        centroids = x[indices].clone()

        # 2. Lặp K-Means bằng PyTorch thuần túy (Siêu tốc trên GPU)
        for _ in range(self.max_iters):
            # Tính khoảng cách Euclidean bình phương: D(x, c) = ||x||^2 + ||c||^2 - 2xc^T
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            c_sq = (centroids ** 2).sum(dim=-1).unsqueeze(0)
            dists = x_sq + c_sq - 2 * torch.matmul(x, centroids.transpose(0, 1)) # [N, K]

            # Gán mỗi patch vào cụm gần nhất
            labels = torch.argmin(dists, dim=-1) # [N]

            # Cập nhật tâm cụm mới bằng cách lấy trung bình (Sử dụng scatter_add để tối ưu vector)
            ones = torch.ones(N, 1, device=x.device)
            counts = torch.zeros(k, 1, device=x.device).scatter_add_(0, labels.unsqueeze(1), ones).clamp(min=1e-9)
            sum_features = torch.zeros(k, D, device=x.device).scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
            centroids = sum_features / counts

        # 3. Kết thúc lặp: Tìm K patch gốc có vị trí gần K tâm cụm nhất
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        c_sq = (centroids ** 2).sum(dim=-1).unsqueeze(0)
        final_dists = x_sq + c_sq - 2 * torch.matmul(x, centroids.transpose(0, 1)) # [N, K]
        
        closest_indices = torch.argmin(final_dists, dim=0)
        
        # Đảm bảo không bị trùng lặp index nếu 2 cụm co lại cùng 1 điểm
        return torch.unique(closest_indices)

    def forward(self, x, **kwargs):
        is_batched = False
        if x.dim() == 3 and x.size(0) == 1:
            is_batched = True
            x = x.squeeze(0)

        # Lấy K index đại diện bằng K-Means
        selected_indices = self._get_kmeans_indices(x, self.num_selected)
        
        # Sắp xếp lại thứ tự không gian
        selected_indices, _ = torch.sort(selected_indices)
        x_selected = x[selected_indices]

        if is_batched:
            x_selected = x_selected.unsqueeze(0)

        return self.mil(x_selected, **kwargs)