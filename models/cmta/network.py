import numpy as np

import torch
import torch.nn as nn
import math
from .util import initialize_weights
from .util import NystromAttention
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention


# ==================================================================================

class FFNExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FFNExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class MoE(nn.Module):
    def __init__(self, input_dim=512, num_experts=4, k=2):
        super(MoE, self).__init__()
        self.k = k
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [FFNExpert(input_dim, input_dim) for _ in range(num_experts)])  # Updated to have input_dim output

    def forward(self, x):
        # x shape: [B, N, input_dim]
        B, N, input_dim = x.shape

        # Reshape x to [B*N, input_dim] for processing
        x_reshaped = x.view(B * N, input_dim)
        gate_scores = self.gate(x_reshaped)  # [B*N, num_experts]
        topk_scores, topk_indices = gate_scores.view(B, N, -1).topk(self.k, dim=2)  # [B, N, k]

        # Initialize the output tensor
        expert_outputs = torch.zeros(B, N, self.k, input_dim,
                                     device=x.device)  # Ensure output dimensions match input_dim

        # Apply the selected experts to the input
        for b in range(B):
            for n in range(N):
                for i, idx in enumerate(topk_indices[b, n]):
                    expert_outputs[b, n, i] = self.experts[idx](x[b, n].unsqueeze(0))

        weights = torch.softmax(topk_scores, dim=2).unsqueeze(-1)  # [B, N, k, 1]
        output = (weights * expert_outputs).sum(dim=2)  # [B, N, input_dim]

        output = output + x

        return output


# =========================================================================================


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, num_experts=4, k=2):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )
        self.moe = MoE(input_dim=dim, num_experts=num_experts, k=k)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512, num_experts=4, k=2):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)

        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->MoE layer
        #   h = self.moe(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        # h = self.norm(h)
        print("P: ",h.shape)
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512, num_experts=4, k=2):
        super(Transformer_G, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.moe = MoE(input_dim=feature_dim, num_experts=num_experts, k=k)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        # h = self.moe(h)

        h = self.layer1(h)  # [B, N, 512]
        # ---->MoE layer
        h = self.moe(h)  # [B, N, 512]
        # ---->PPEG
        # h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        # ---->MoE layer
        h = self.moe(h)  # [B, N, 512]
        print("G:",h.shape)
        return h[:, 0], h[:, 1:]


class token_selection(nn.Module):
    def __init__(self):
        super(token_selection, self).__init__()
        self.MLP_f = nn.Linear(256, 128)
        self.MLP_s= nn.Linear(256, 256)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, start_patch_token, cls_token):
        half_token_patch = self.MLP_f(start_patch_token)
        half_token_cls = self.MLP_f(cls_token)
        half_token_cls = half_token_cls.unsqueeze(1)
        half_token_cls = half_token_cls.repeat(1, start_patch_token.size(1), 1)  # Corrected this line
        patch_token = torch.cat([half_token_cls, half_token_patch], dim=2)
        patch_token = self.MLP_s(patch_token)
        _patch_token = self.softmax(patch_token)
        topk_values, topk_indices = torch.topk(_patch_token, math.ceil(start_patch_token.size(1)/2), dim=1)
        final_token = torch.gather(start_patch_token, 1, topk_indices.squeeze(1))  # Squeeze the last dimension here

        return final_token



from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
from torch import nn
from hypll import nn as hnn


manifold = PoincareBall(c=Curvature(requires_grad=True))

class CMTA(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat", model_size="small",alpha=0.5,beta=0.5,tokenS="both"):
        super(CMTA, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.alpha=alpha
        self.beta=beta
        self.tokenS=tokenS
        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
        # Decoder
        self.pathomics_decoder = Transformer_P(feature_dim=hidden[-1])

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        # G->P Attention
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)

        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])
        # Decoder
        self.genomics_decoder = Transformer_G(feature_dim=hidden[-1])

        self.hyperbolic_fc1 = hnn.HLinear(in_features=hidden[-1] * 2, out_features=hidden[-1], manifold=manifold)
        self.hyperbolic_fc2 = hnn.HLinear(in_features=hidden[-1], out_features=hidden[-1], manifold=manifold)
        self.hyperbolic_relu = hnn.HReLU(manifold=manifold)
        self.token_selection = token_selection()


        # Classification Layer
        if self.fusion == "Aconcat" or self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "fineCoarse":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        elif self.fusion == "hyperbolic":
            self.hyperbolic_mm = nn.Sequential(
                self.hyperbolic_fc1,
                self.hyperbolic_relu,
                self.hyperbolic_fc2,
                self.hyperbolic_relu,
                self.hyperbolic_fc2
            )
        elif self.fusion == "hyperbolic":
            self.hyperbolic_mm = nn.Sequential(
                self.hyperbolic_fc1,
                self.hyperbolic_relu,
                self.hyperbolic_fc2,
                self.hyperbolic_relu,
                self.hyperbolic_fc2
            )
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]

        # Enbedding
        # genomics embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, 1024]
        # pathomics embedding
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)
        print("genomics_features.shape: ",genomics_features.shape)
        print("pathomics_features.shape:",pathomics_features.shape)
        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        print("cls_token_pathomics_encoder.shape: ",cls_token_pathomics_encoder.shape)
        print("cls_token_genomics_encoder.shape: ",cls_token_genomics_encoder.shape)
        print("patch_token_pathomics_encoder.shape: ",patch_token_pathomics_encoder.shape)
        print("patch_token_genomics_encoder.shape: ",patch_token_genomics_encoder.shape)
        # cross-omics attention

        #=============== token selection;

        if self.tokenS=="both":
            patch_token_pathomics_encoder=self.token_selection(patch_token_pathomics_encoder, cls_token_pathomics_encoder)
            patch_token_genomics_encoder=self.token_selection(patch_token_genomics_encoder, cls_token_genomics_encoder)
        elif self.tokenS=="P":
            patch_token_pathomics_encoder=self.token_selection(patch_token_pathomics_encoder, cls_token_pathomics_encoder)
        elif self.tokenS=="G":
            patch_token_genomics_encoder=self.token_selection(patch_token_genomics_encoder, cls_token_genomics_encoder)

        # =============== token selection;
        print("===========")
        print("patch_token_pathomics_encoder.shape: ", patch_token_pathomics_encoder.shape)
        print("patch_token_genomics_encoder.shape: ", patch_token_genomics_encoder.shape)
        print("===========")
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        # decoder
        print(" pathomics_in_genomics: " ,pathomics_in_genomics.shape)
        print(" genomics_in_pathomics: " ,genomics_in_pathomics.shape)


        # pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        # genomics decoder
        cls_token_genomics_decoder, _ = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens
        # cls_token_pathomics_decoder, _ = self.genomics_decoder(patch_token_pathomics_encoder )
        # cls_token_genomics_decoder, _ = self.genomics_decoder(patch_token_genomics_encoder)
        # fusion
        # print("cls_token_pathomics_encoder", cls_token_pathomics_encoder.shape)
        # print("cls_token_genomics_encoder", cls_token_genomics_encoder.shape)
        print("cls_token_pathomics_decoder", cls_token_pathomics_decoder.shape)
        print("cls_token_genomics_decoder", cls_token_genomics_decoder.shape)
        if self.fusion == "concat":
            fusion = self.mm(
                torch.concat(
                    (
                        (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                        (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ),
                    dim=1,
                )
            )  # take cls token to make prediction
        elif self.fusion == "Aconcat":
            fusion = self.mm(
                torch.concat(
                    (
                        (1 - self.alpha) * (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                        self.alpha * (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ),
                    dim=1,
                )
            )  #
        elif self.fusion == "fineCoarse":
            fusion_coarse = self.mm(
                torch.concat(
                    (
                        cls_token_pathomics_encoder,
                        cls_token_genomics_encoder,
                    ),
                    dim=1
                )
            )
            fusion_fine = self.mm(
                torch.concat(
                    (
                        cls_token_pathomics_decoder,
                        cls_token_genomics_decoder,
                    ),
                    dim=1
                )
            )
            fusion=self.beta * fusion_fine + (1-self.beta) * fusion_coarse
        elif self.fusion == "bilinear":
            fusion = self.mm(
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
            )  # take cls token to make prediction
        elif self.fusion == "hyperbolic":
            # Step 1: Compute the average of pathomics encoder and decoder cls tokens
            # print("hyperbolic")
            pathomics_avg = (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2
            genomics_avg = (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2

            # Step 2: Concatenate the averaged features from pathomics and genomics
            concatenated_features = torch.cat((pathomics_avg, genomics_avg), dim=1)

            # Step 3: Wrap the concatenated features as a tangent vector on the manifold
            tangent_features = TangentTensor(data=concatenated_features, man_dim=1, manifold=manifold)

            # Step 4: Map the tangent vector to the manifold using the exponential map
            hy_features = manifold.expmap(tangent_features)

            # Step 5: Apply hyperbolic matrix multiplication to map features within the hyperbolic space
            fusion_hy= self.hyperbolic_mm(hy_features)

            # Define the origin point on the manifold for logmap
            # origin = torch.zeros_like(fusion_hy.tensor)  # Assuming the origin is a zero tensor of the same shape

            # Map the fusion tensor back to Euclidean space using the logarithmic map
            # log_mapped_fusion = manifold.logmap(origin, fusion_hy)

            # Step 7: Retrieve the tensor from the log-mapped structure
            fusion = fusion_hy.tensor
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))
        # fusion=( cls_token_genomics_decoder +  cls_token_genomics_encoder) / 2
        # fusion = (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2
        # predict
        # predict
        logits = self.classifier(fusion)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, cls_token_pathomics_encoder, cls_token_pathomics_decoder, cls_token_genomics_encoder, cls_token_genomics_decoder
