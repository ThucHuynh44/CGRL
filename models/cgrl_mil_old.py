
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# HÀM KHỞI TẠO TRỌNG SỐ
# ==============================================================================
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
# 1. BỘ CHỌN LỌC VECTOR (Content-Aware Compression + Concept Similarity)
# ==============================================================================
class ConceptSelector(nn.Module):
    """
    Module chọn lọc vector nén (CAC) kết hợp InfoNCE (Sim Loss).
    Đầu vào: N patch [N, input_dim]
    Đầu ra: K patch gốc tốt nhất [K, input_dim] cùng với dictionary chứa sim_loss và cac_loss
    """
    def __init__(self, input_dim=512, compress_dim=128, n_classes=2, concept_path=None, device='cuda'):
        super(ConceptSelector, self).__init__()
        self.n_classes = n_classes
        self.device = device
        
        # --- 1. MODULE NÉN (CAC) ---
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, compress_dim)
        )
        self.compressor.apply(initialize_weights)
        
        # Các lớp Linear để tính điểm Attention giả lập cho CAC Loss
        self.W_f = nn.Linear(input_dim, 1, bias=False)
        self.W_v = nn.Linear(compress_dim, 1, bias=False)
        self.W_f.apply(initialize_weights)
        self.W_v.apply(initialize_weights)
        
        # --- 2. MODULE SIM LOSS (InfoNCE) ---
        # Đảm bảo bạn đã có file models/infonce.py
        from models.infonce import InfoNCE
        self.info_nce_loss = InfoNCE(temperature=0.1, reduction='mean', negative_mode='unpaired')
        
        # --- 3. CONCEPTS ---
        self.concepts = None
        if concept_path is not None:
            rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            # Load Concepts ở không gian gốc
            self._load_concepts(concept_path, input_dim)
            
            torch.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

    def _load_concepts(self, path, input_dim):
        """Chỉ load concepts gốc và giữ nguyên không gian input_dim (ví dụ 512)"""
        try:
            raw_concepts = torch.load(path, map_location=self.device) 
            if isinstance(raw_concepts, list):
                raw_concepts = raw_concepts[:self.n_classes]
            
            processed_concepts = []
            for c in raw_concepts:
                if isinstance(c, torch.Tensor):
                    c = c.to(self.device)
                if c.dim() > 1:
                    c = c.mean(dim=0) 
                processed_concepts.append(c)
            
            # Khởi tạo Parameter gốc [C, input_dim] để có thể học được
            self.concepts = nn.Parameter(torch.stack(processed_concepts, dim=0))

        except Exception as e:
            print(f"WARNING: Could not load concepts. Error: {e}")
            self.concepts = None

    # def forward(self, x, label=None, lambda_sym=1.0, lambda_cross=1.0, topk_pos=20, topk_neg=20):
    #     # x shape: [N, input_dim]
    #     results_dict = {
    #         'sim_loss': torch.tensor(0.0, device=x.device),
    #         'cac_loss': torch.tensor(0.0, device=x.device)
    #     }

    #     if self.concepts is None:
    #         return x, results_dict

    #     # --- BƯỚC 1: NÉN CẢ ĐẶC TRƯNG LẪN CONCEPTS VỀ CÙNG KHÔNG GIAN ---
    #     v_ins = self.compressor(x) # [N, compress_dim]
    #     c_ins = self.compressor(self.concepts) # [C, compress_dim]
        
    #     # --- BƯỚC 2: TÍNH CAC LOSS ---
    #     # Tính điểm attention giả lập để ép mạng nén giữ lại thông tin quan trọng
    #     score_f = self.W_f(x).squeeze(-1) # [N]
    #     score_v = self.W_v(v_ins).squeeze(-1) # [N]
        
    #     a_f = F.softmax(score_f, dim=0)
    #     a_v = F.softmax(score_v, dim=0)
        
    #     # Dùng detach() để không backprop qua x gốc, chỉ cập nhật mạng nén
    #     cac_loss = F.mse_loss(a_v, a_f.detach())
    #     results_dict['cac_loss'] = cac_loss

    #     # --- BƯỚC 3: TÍNH ĐỘ TƯƠNG ĐỒNG BẰNG VECTOR NÉN ---
    #     V_norm = F.normalize(v_ins, dim=-1)
    #     T_norm = F.normalize(c_ins, dim=-1)
    #     S = V_norm @ T_norm.t() # [N, C]

    #     # --- BƯỚC 4: LỰA CHỌN VECTOR (NGƯỠNG ĐỘNG MEAN) ---
    #     max_sim_per_instance, _ = S.max(dim=-1) 
    #     threshold = max_sim_per_instance.mean()
    #     selected_indices = torch.nonzero(max_sim_per_instance > threshold).squeeze(-1)
        
    #     # Fallback an toàn nếu không có patch nào qua ngưỡng
    #     if selected_indices.numel() == 0:
    #         _, selected_indices = torch.topk(max_sim_per_instance, 1, dim=0)

    #     # TRẢ VỀ ĐẶC TRƯNG NGUYÊN BẢN (CHƯA NÉN) CHO MẠNG MIL
    #     x_selected = x[selected_indices]

    #     # --- BƯỚC 5: TÍNH SIM LOSS (ĐỂ CẬP NHẬT MẠNG NÉN VÀ CONCEPTS) ---
    #     if label is not None:
    #         # Truyền v_ins và c_ins vào để tính toán InfoNCE
    #         sim_loss = self._calculate_sim_loss(v_ins, c_ins, S, label, lambda_sym, lambda_cross, topk_pos, topk_neg)
    #         results_dict['sim_loss'] = sim_loss

    #     return x_selected, results_dict

    def forward(self, x, label=None, lambda_sym=1.0, lambda_cross=1.0, topk_pos=20, topk_neg=20):
        # --- BƯỚC 0: XỬ LÝ CHIỀU BATCH ---
        # Kiểm tra xem x có dạng [1, N, D] hay không. Nếu có thì tạm thời lột bỏ chiều 1 đi.
        num_selected=300
        is_batched = False
        if x.dim() == 3 and x.size(0) == 1:
            is_batched = True
            x = x.squeeze(0) # Đưa về [N, D] để dễ tính toán
            
        results_dict = {
            'sim_loss': torch.tensor(0.0, device=x.device),
            'cac_loss': torch.tensor(0.0, device=x.device)
        }

        if self.concepts is None:
            if is_batched: x = x.unsqueeze(0) # Lắp lại chiều nếu lúc đầu đã lột
            return x, results_dict

        # --- BƯỚC 1: NÉN CẢ ĐẶC TRƯNG LẪN CONCEPTS ---
        v_ins = self.compressor(x) # [N, compress_dim]
        c_ins = self.compressor(self.concepts) # [C, compress_dim]
        
        # --- BƯỚC 2: TÍNH CAC LOSS ---
        score_f = self.W_f(x).squeeze(-1) # [N]
        score_v = self.W_v(v_ins).squeeze(-1) # [N]
        
        # Quan trọng: Phải là dim=-1 (chiều cuối cùng) để softmax lấy xác suất theo các patch N
        a_f = F.softmax(score_f, dim=-1)
        a_v = F.softmax(score_v, dim=-1)
        
        cac_loss = F.mse_loss(a_v, a_f.detach())* (x.size(0) ** 2)
        results_dict['cac_loss'] = cac_loss

        # --- BƯỚC 3: TÍNH ĐỘ TƯƠNG ĐỒNG BẰNG VECTOR NÉN ---
        V_norm = F.normalize(v_ins, dim=-1)
        T_norm = F.normalize(c_ins, dim=-1)
        S = V_norm @ T_norm.t() # [N, C]

        # --- BƯỚC 4: LỰA CHỌN VECTOR (NGƯỠNG ĐỘNG MEAN) ---
        max_sim_per_instance, _ = S.max(dim=-1) 
        k_select = min(num_selected, x.size(0))
        _, selected_indices = torch.topk(max_sim_per_instance, k_select, dim=0)
        selected_indices, _ = torch.sort(selected_indices)
        
        # Trích xuất các patch tốt nhất
        x_selected = x[selected_indices] # [K, D]

        # --- BƯỚC 5: TÍNH SIM LOSS ---
        if label is not None:
            sim_loss = self._calculate_sim_loss(v_ins, c_ins, S, label, lambda_sym, lambda_cross, topk_pos, topk_neg)
            results_dict['sim_loss'] = sim_loss

        # --- BƯỚC 6: LẮP LẠI CHIỀU BATCH ---
        # Nếu x lúc đầu vào là [1, N, D], ta trả về [1, K, D] để không làm vỡ mạng MIL phía sau
        if is_batched:
            x_selected = x_selected.unsqueeze(0)

        return x_selected, results_dict

    def _calculate_sim_loss(self, X, c_ins, S, label, lambda_sym, lambda_cross, topk_pos, topk_neg):
        """
        X: v_ins [N, compress_dim]
        c_ins: Concepts nén [C, compress_dim]
        S: Ma trận tương đồng [N, C]
        """
        T = c_ins 
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


# ==============================================================================
# 3. KẾT HỢP (Wrapper Model Plug-and-Play)
# ==============================================================================
class CGRLMIL(nn.Module):
    """
    Kết nối Selector và BẤT KỲ mạng MIL nào. 
    Dữ liệu sẽ qua bộ chọn lọc để giảm từ N -> K vector (ngưỡng động) trước khi vào mạng MIL.
    """
    def __init__(self, base_mil_model, input_dim=512, compress_dim=128, n_classes=2, concept_path=None):
        super().__init__()
        
        self.selector = ConceptSelector(
            input_dim=input_dim, 
            compress_dim=compress_dim,
            n_classes=n_classes, 
            concept_path=concept_path
            # Bỏ num_selected vì dùng ngưỡng mean
        )
        
        # Gắn mạng MIL bất kỳ được truyền từ ngoài vào
        self.mil = base_mil_model

    def forward(self, x, label=None, **kwargs):
        # 1. Lọc và lấy subset vector tốt nhất (cùng với dict chứa các loại loss)
        x_selected, results_dict = self.selector(x, label)
        
        # 2. Truyền x_selected (đã được lọc nhưng chưa bị nén) vào mạng MIL cốt lõi
        x_out, Y_prob, Y_hat, mil_results, feature = self.mil(x_selected)
        
        # (Tuỳ chọn) Nếu mạng MIL có loss riêng, gộp vào results_dict
        # if mil_results is not None:
        #     results_dict.update(mil_results)
        
        return x_out, Y_prob, Y_hat, results_dict, feature