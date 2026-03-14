from typing import List

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import spacy


class TransformerSimilarity:
    """
    使用 HuggingFace Transformer 模型计算两组句子之间的相似度/距离矩阵，
    并支持从句子中提取名词/动词，计算与给定词语列表的命中重叠数，
    同时可返回每个命中的详细信息（关键词、匹配词、相似度）。
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        初始化模型、tokenizer 和 spaCy 词性标注工具。

        Args:
            model_name: HuggingFace 模型名称
            device: 计算设备（自动选择 GPU/CPU）
            spacy_model: spaCy 英文模型名称，需预先安装（`python -m spacy download en_core_web_sm`）
        """
        # 加载 Transformer 模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

        # 加载 spaCy 模型（用于词性标注）
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise OSError(
                f"spaCy 模型 '{spacy_model}' 未找到。请运行: python -m spacy download {spacy_model}"
            )

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling，考虑 attention mask。"""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        将句子列表编码为归一化的句向量。

        Args:
            sentences: 句子列表
            batch_size: 批处理大小

        Returns:
            归一化句向量矩阵，形状 (len(sentences), hidden_dim)
        """
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def extract_keywords(self, sentence: str) -> List[str]:
        """
        使用 spaCy 提取句子中的名词和动词（词形还原后的形式）。

        Args:
            sentence: 输入句子

        Returns:
            关键词列表（动词和名词的原形）
        """
        doc = self.nlp(sentence)
        keywords = list(set([token.lemma_ for token in doc if token.pos_ in {"VERB", "NOUN"}]))
        return keywords

    @torch.inference_mode()
    def compute_overlap(
        self,
        sentences: List[str],
        words: List[str],
        threshold: float = 0.7,
        batch_size: int = 32,
        return_meta: bool = False,
    ) -> Union[
        List[Tuple[int, int, int]],
        Tuple[List[Tuple[int, int, int]], List[List[Tuple[str, str, float]]]],
    ]:
        """
        计算每个句子的名词/动词与给定词语列表的命中重叠数，并可选择返回命中详情。

        流程：
        1. 提取每个句子的关键词（名词和动词），并记录每个关键词对应的句子索引。
        2. 将所有关键词扁平化为一个列表，一次性编码，同时编码词语列表。
        3. 计算关键词与词语的相似度矩阵，找出每个关键词的最大相似度及对应词语。
        4. 统计每个句子中命中（相似度 ≥ threshold）的关键词数量。
        5. 若 return_meta=True，同时返回每个句子中命中的关键词、匹配的词语和相似度。

        Args:
            sentences: 句子列表（N 个）
            words: 词语列表（M 个）
            threshold: 相似度阈值，超过该值认为命中
            batch_size: 编码时的批处理大小
            return_meta: 是否返回每个句子命中的详细信息

        Returns:
            如果 return_meta=False，返回 results: List[Tuple[int, int, int]]
                每个元组对应一个句子：(命中关键词数量, 该句子提取的关键词总数, 词语列表长度)
            如果 return_meta=True，返回 (results, meta_data)
                results: 同上
                meta_data: List[List[Tuple[str, str, float]]]，每个句子包含命中的 (keyword, matched_word, similarity)
        """
        words = [w.replace("'", "") for w in words]
        # Step 1: 提取所有句子的关键词，并记录所属句子索引
        all_keywords = []  # 扁平关键词列表
        sent_keyword_counts = []  # 每个句子的关键词数量
        sent_indices = []  # 每个扁平关键词对应的句子索引

        for idx, sent in enumerate(sentences):
            kw_list = self.extract_keywords(sent)
            sent_keyword_counts.append(len(kw_list))
            all_keywords.extend(kw_list)
            sent_indices.extend([idx] * len(kw_list))

        # 如果没有关键词，直接返回全零结果（避免编码空列表）
        if not all_keywords:
            empty_results = [(0, 0, len(words)) for _ in sentences]
            if return_meta:
                empty_meta = [[] for _ in sentences]
                return empty_results, empty_meta
            else:
                return empty_results

        # Step 2: 编码扁平关键词列表和词语列表
        kw_embeddings = self.encode(all_keywords, batch_size=batch_size)  # (K, D)
        word_embeddings = self.encode(words, batch_size=batch_size)  # (M, D)

        # Step 3: 计算相似度矩阵 (K, M)
        sim_matrix = torch.mm(
            kw_embeddings, word_embeddings.T
        )  # 点积 = 余弦相似度（已归一化）

        # 找出每个关键词的最大相似度及对应词语索引
        max_sim, max_idx = sim_matrix.max(dim=1)  # (K,), (K,)
        max_sim = max_sim.cpu()
        max_idx = max_idx.cpu()

        # Step 4: 统计命中数并收集元数据
        sentence_hits = torch.zeros(len(sentences), dtype=torch.int)
        if return_meta:
            meta_data = [[] for _ in sentences]

        for i, (sim_val, idx_word) in enumerate(zip(max_sim, max_idx)):
            if sim_val >= threshold:
                sent_idx = sent_indices[i]
                sentence_hits[sent_idx] += 1
                if return_meta:
                    keyword = all_keywords[i]
                    matched_word = words[idx_word]
                    meta_data[sent_idx].append((keyword, matched_word, sim_val.item()))

        # Step 5: 构建结果列表
        results = []
        for i in range(len(sentences)):
            hits = int(sentence_hits[i])
            num_keywords = sent_keyword_counts[i]
            results.append((hits, num_keywords, len(words)))

        if return_meta:
            return results, meta_data
        else:
            return results


class RewardModel:
    def __init__(
        self,
        format_output,
        model_name,
        length_reward_ratio=0,
        object_reward_ratio=0,
        keyframe_reward_ratio=0,
    ):
        self.length_reward_ratio = length_reward_ratio
        self.object_reward_ratio = object_reward_ratio
        self.keyframe_reward_ratio = keyframe_reward_ratio
        self.format_output = format_output
        if object_reward_ratio != 0:
            self.sim_calc = TransformerSimilarity(model_name)

    def __call__(
        self, completions: List, truth: List[str], input_object, input_keyframe, completion_length, **kwargs
    ):
        """返回(format_reward, correct_reward, object_reward, keyframe_reward, length_reward)"""

        rewards = []
        for completion, sol, obj, keyframe, length in zip(
            completions, truth, input_object, input_keyframe, completion_length
        ):
            reward = [0 for i in range(5)]
            content = completion[0]["content"]
            res = self.format_output(content)
            if res is None:
                reward[0] = -0.5  # 出现错误
            else:
                reward[1] = 0 if res["answer"] != sol else 1

                if self.object_reward_ratio != 0:
                    words = obj
                    object_reward = 0
                    if len(words) != 0:
                        overlap_results = self.sim_calc.compute_overlap(
                            res["reasoning"], words
                        )
                        for hits, num_kw, m_len in overlap_results:
                            object_reward += hits * 0.1
                        reward[2] = reward[1] * self.object_reward_ratio * np.clip(object_reward, 0, 1)

                if self.keyframe_reward_ratio != 0:
                    s1 = set(keyframe)
                    s2 = set(res["keyframes"])
                    IoU = len(s1.intersection(s2)) / len(s1.union(s2))
                    reward[3] = self.keyframe_reward_ratio * IoU
                if self.length_reward_ratio != 0:
                    current_len = sum(len(r) for r in res["reasoning"])
                    length_reward = np.exp(-((length - 120)**2) / (2 * 200**2)) 
                    reward[4] = self.length_reward_ratio * length_reward
            rewards.append(reward)
        return rewards
