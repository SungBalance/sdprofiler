import os
import sys
import warnings
import re
import argparse
from typing import List, Dict, Tuple, Optional
import json
import time
from collections import defaultdict

# 기본 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, set_seed
)
import torch


plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class LocalLLMSentenceAnalyzer:
    """로컬 LLM 문장 길이 분석기 클래스"""
    
    def __init__(self, verbose: bool = True):
        """초기화"""
        self.dataset = None
        self.sentences_data = []
        self.requests_data = []
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.current_model_name = None
        self.verbose = verbose
        
        # 추천 모델 목록
        self.recommended_models = {
            "korean": [
                "skt/kogpt2-base-v2",  # 125M
                "kakaobrain/kogpt",    # 6B
                "nlpai-lab/kullm-polyglot-5.8b-v2",  # 5.8B
            ],
            "english_small": [
                "microsoft/DialoGPT-small",    # 117M
                "microsoft/DialoGPT-medium",   # 345M
                "facebook/blenderbot-400M-distill",  # 400M
                "google/flan-t5-small",        # 80M
                "google/flan-t5-base",         # 250M
                "EleutherAI/gpt-neo-125M",     # 125M
            ],
            "english_large": [
                "microsoft/DialoGPT-large",    # 762M
                "facebook/blenderbot-1B-distill",   # 1B
                "google/flan-t5-large",        # 780M
                "EleutherAI/gpt-neo-1.3B",     # 1.3B
                "microsoft/GODEL-v1_1-base-seq2seq",  # 220M
            ]
        }
    
    def log(self, message: str):
        """로그 출력 (verbose 모드일 때만)"""
        if self.verbose:
            print(message)
    
    def load_local_model(self, model_name: str, device: str = "auto", use_cache: bool = True):
        """로컬 LLM 모델 로드"""
        self.log(f"🔄 모델 로딩 중: {model_name}")
        
        try:
            # 디바이스 설정
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.log(f"🖥️  사용 디바이스: {device}")
            
            # 메모리 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                del self.tokenizer
                del self.pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 토크나이저 로드
            self.log("📝 토크나이저 로딩...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                cache_dir="./model_cache" if use_cache else None
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 설정
            model_kwargs = {
                "cache_dir": "./model_cache" if use_cache else None,
                "torch_dtype": torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            # T5 계열 모델은 Seq2Seq, 나머지는 CausalLM
            if "t5" in model_name.lower() or "godel" in model_name.lower():
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, **model_kwargs
                ).to(device)
                
                # Seq2Seq 모델용 파이프라인
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if device == "cuda" else -1,
                    torch_dtype=model_kwargs["torch_dtype"]
                )
                self.log("📋 모델 타입: Text-to-Text Generation (Seq2Seq)")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, **model_kwargs
                ).to(device)
                
                # CausalLM 모델용 파이프라인
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if device == "cuda" else -1,
                    torch_dtype=model_kwargs["torch_dtype"]
                )
                self.log("모델 타입: Text Generation (CausalLM)")
            
            self.current_model_name = model_name
            
            # 모델 정보 출력
            num_params = self.model.num_parameters()
            self.log(f"🧠 모델 파라미터 수: {num_params:,} ({num_params/1e6:.1f}M)")
            
            # GPU 메모리 사용량 (GPU 사용 시)
            if device == "cuda" and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.log(f"💾 GPU 메모리 사용량: {memory_used:.1f}GB / {memory_total:.1f}GB")
            
            self.log(f"✅ 모델 로딩 완료!")
            return True
            
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            return False
    
    def load_dataset(self, dataset_name: str, subset: str = None, split: str = "train", 
                    sample_size: int = 1000, language: str = None):
        """데이터셋 로드"""
        if not DATASETS_AVAILABLE:
            print("datasets 라이브러리가 설치되지 않았습니다.")
            return False
        
        self.log(f"📊 데이터셋 로딩 중: {dataset_name}")
        
        try:
            # 데이터셋 로드
            load_kwargs = {}
            if subset:
                load_kwargs['name'] = subset
            
            self.log("🔄 데이터셋 다운로드 중...")
            self.dataset = load_dataset(dataset_name, split=split, **load_kwargs)
            
            self.log(f"📥 원본 데이터셋 크기: {len(self.dataset):,}개")
            
            # 언어 필터링 (해당하는 경우)
            if language and 'lang' in self.dataset.column_names:
                original_size = len(self.dataset)
                self.dataset = self.dataset.filter(lambda x: x.get('lang') == language)
                self.log(f"🌐 언어 필터링 ({language}): {len(self.dataset):,}개 (원본: {original_size:,}개)")
            
            # 샘플 크기 제한
            if len(self.dataset) > sample_size:
                indices = np.random.choice(len(self.dataset), sample_size, replace=False)
                self.dataset = self.dataset.select(indices)
                self.log(f"📦 랜덤 샘플링: {sample_size:,}개 데이터 선택")
            
            self.log(f"✅ 최종 데이터셋 크기: {len(self.dataset):,}개 항목")
            
            return True
            
        except Exception as e:
            print(f"데이터셋 로딩 실패: {e}")
            return False
    
    def generate_responses(self, prompts: List[str], max_length: int = 100, 
                          num_return_sequences: int = 1, temperature: float = 0.7,
                          top_p: float = 0.9, do_sample: bool = True) -> List[str]:
        """로컬 LLM으로 응답 생성"""
        if not self.pipeline:
            print("먼저 모델을 로드해주세요")
            return []
        
        self.log(f"🎯 {len(prompts)}개 프롬프트로 응답 생성 중...")
        
        responses = []
        set_seed(42)  # 재현 가능한 결과
        
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            try:
                if self.verbose:
                    print(f"📝 {i+1}/{len(prompts)}: 생성 중... ", end="", flush=True)
                
                # 생성 설정
                generation_kwargs = {
                    "max_length": max_length,
                    "num_return_sequences": num_return_sequences,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                # 모델 타입에 따른 생성
                if "t5" in str(type(self.model)).lower() or "godel" in str(type(self.model)).lower():
                    # Seq2Seq 모델
                    result = self.pipeline(prompt, **generation_kwargs)
                    generated_text = result[0]['generated_text']
                else:
                    # CausalLM 모델
                    generation_kwargs["max_length"] = len(self.tokenizer.encode(prompt)) + max_length
                    result = self.pipeline(prompt, **generation_kwargs)
                    # 원본 프롬프트 제거
                    generated_text = result[0]['generated_text'][len(prompt):].strip()
                
                responses.append(generated_text)
                
                if self.verbose:
                    print("완료")
                
            except Exception as e:
                if self.verbose:
                    print(f"실패: {e}")
                responses.append("생성 실패")
        
        total_time = time.time() - start_time
        self.log(f"✅ 전체 응답 생성 완료! (총 {total_time:.1f}초)")
        
        return responses
    
    def split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분할"""
        sentence_pattern = r'[.!?]+\s+|[。！？]+\s*|[\n\r]+'
        sentences = re.split(sentence_pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 1]
        return sentences
    
    def calculate_sentence_lengths(self, sentences: List[str], unit='characters') -> List[int]:
        """문장 길이 계산"""
        if unit == 'characters':
            return [len(s) for s in sentences]
        elif unit == 'words':
            return [len(s.split()) for s in sentences]
        elif unit == 'tokens':
            if self.tokenizer:
                lengths = []
                for s in sentences:
                    try:
                        tokens = self.tokenizer.encode(s, add_special_tokens=False)
                        lengths.append(len(tokens))
                    except:
                        lengths.append(len(s.split()))
                return lengths
            else:
                return [len(s.split()) + len(re.findall(r'[^\w\s]', s)) for s in sentences]
        else:
            raise ValueError("unit은 'characters', 'words', 또는 'tokens' 중 하나여야 합니다.")
    
    def analyze_existing_data(self, text_field: str = None, unit='characters'):
        """기존 데이터셋 분석"""
        if not self.dataset:
            print("먼저 데이터셋을 로드해주세요")
            return None
        
        self.log(f"📊 기존 데이터셋 분석 ({unit} 기준)")
        
        all_sentences = []
        request_sentence_counts = []
        request_avg_lengths = []
        processed_count = 0
        
        # 데이터 처리
        for i, item in enumerate(self.dataset):
            # 텍스트 필드 자동 찾기
            text = ""
            if text_field and text_field in item:
                text = item[text_field]
            else:
                # 자동 텍스트 필드 탐지
                for field_name in ["text", "content", "response", "answer", "message", "output"]:
                    if field_name in item and isinstance(item[field_name], str):
                        text = item[field_name]
                        break
                
                # 여전히 없으면 첫 번째 긴 문자열 필드 사용
                if not text:
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 10:
                            text = value
                            break
            
            if text and isinstance(text, str) and text.strip():
                sentences = self.split_into_sentences(text)
                if sentences:
                    sentence_lengths = self.calculate_sentence_lengths(sentences, unit)
                    
                    all_sentences.extend(sentence_lengths)
                    request_sentence_counts.append(len(sentences))
                    if sentence_lengths:
                        request_avg_lengths.append(np.mean(sentence_lengths))
                    
                    processed_count += 1
            
            # 진행 상황 출력
            if self.verbose and (i + 1) % 100 == 0:
                print(f"🔄 처리 중: {i+1}/{len(self.dataset)} ({processed_count}개 유효)")
        
        # 통계 계산
        if all_sentences:
            stats = {
                'total_sentences': len(all_sentences),
                'total_requests': len(request_sentence_counts),
                'processed_items': processed_count,
                'avg_sentence_length': np.mean(all_sentences),
                'median_sentence_length': np.median(all_sentences),
                'std_sentence_length': np.std(all_sentences),
                'min_sentence_length': min(all_sentences),
                'max_sentence_length': max(all_sentences),
                'avg_sentences_per_request': np.mean(request_sentence_counts),
                'avg_request_avg_length': np.mean(request_avg_lengths) if request_avg_lengths else 0
            }
            
            # 결과 출력
            print(f"\n📈 분석 결과:")
            print(f"  • 처리된 항목: {stats['processed_items']:,} / {len(self.dataset):,}")
            print(f"  • 총 문장 수: {stats['total_sentences']:,}")
            print(f"  • 평균 문장 길이: {stats['avg_sentence_length']:.2f} {unit}")
            print(f"  • 중간값: {stats['median_sentence_length']:.2f} {unit}")
            print(f"  • 표준편차: {stats['std_sentence_length']:.2f} {unit}")
            print(f"  • 범위: {stats['min_sentence_length']} - {stats['max_sentence_length']} {unit}")
            print(f"  • 요청당 평균 문장 수: {stats['avg_sentences_per_request']:.2f}")
            
            # 데이터 저장
            self.sentences_data = all_sentences
            self.requests_data = {
                'sentence_counts': request_sentence_counts,
                'avg_lengths': request_avg_lengths
            }
            
            return stats
        else:
            print("분석할 데이터가 없습니다.")
            return None
    
    def analyze_generated_responses(self, prompts: List[str], unit='characters', 
                                  max_length: int = 100, temperature: float = 0.7,
                                  show_responses: bool = False):
        """로컬 LLM으로 생성한 응답 분석"""
        if not self.pipeline:
            print("먼저 모델을 로드해주세요")
            return None, []
        
        self.log(f"🎯 생성된 응답 분석 ({unit} 기준)")
        
        # 응답 생성
        responses = self.generate_responses(
            prompts, 
            max_length=max_length, 
            temperature=temperature
        )
        
        # 응답 분석
        all_sentences = []
        request_sentence_counts = []
        request_avg_lengths = []
        
        if show_responses:
            print(f"\n📝 생성된 응답들:")
            print("-" * 50)
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            if show_responses:
                print(f"\n{i+1}. 프롬프트: {prompt}")
                print(f"   응답: {response}")
            
            sentences = self.split_into_sentences(response)
            sentence_lengths = self.calculate_sentence_lengths(sentences, unit)
            
            all_sentences.extend(sentence_lengths)
            request_sentence_counts.append(len(sentences))
            if sentence_lengths:
                request_avg_lengths.append(np.mean(sentence_lengths))
        
        # 통계 계산 및 출력
        if all_sentences:
            stats = {
                'total_sentences': len(all_sentences),
                'total_requests': len(prompts),
                'avg_sentence_length': np.mean(all_sentences),
                'median_sentence_length': np.median(all_sentences),
                'std_sentence_length': np.std(all_sentences),
                'min_sentence_length': min(all_sentences),
                'max_sentence_length': max(all_sentences),
                'avg_sentences_per_request': np.mean(request_sentence_counts),
                'avg_request_avg_length': np.mean(request_avg_lengths) if request_avg_lengths else 0
            }
            
            print(f"\n📊 분석 결과:")
            print(f"  • 총 문장 수: {stats['total_sentences']:,}")
            print(f"  • 평균 문장 길이: {stats['avg_sentence_length']:.2f} {unit}")
            print(f"  • 중간값: {stats['median_sentence_length']:.2f} {unit}")
            print(f"  • 표준편차: {stats['std_sentence_length']:.2f} {unit}")
            print(f"  • 범위: {stats['min_sentence_length']} - {stats['max_sentence_length']} {unit}")
            print(f"  • 요청당 평균 문장 수: {stats['avg_sentences_per_request']:.2f}")
            
            # 데이터 저장
            self.sentences_data = all_sentences
            self.requests_data = {
                'sentence_counts': request_sentence_counts,
                'avg_lengths': request_avg_lengths
            }
            
            return stats, responses
        else:
            print("분석할 데이터가 없습니다.")
            return None, responses
    
    def plot_distributions(self, unit='characters', save_path: str = None):
        """분포 시각화"""
        if not self.sentences_data:
            print("분석할 데이터가 없습니다.")
            return
        
        self.log(f"📊 분포 시각화 생성 중...")
        
        # 스타일 설정
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'LLM 문장 길이 분석 결과 ({unit})', fontsize=16, fontweight='bold')
        
        # 색상 팔레트
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 1. 전체 문장 길이 분포 (히스토그램)
        axes[0, 0].hist(self.sentences_data, bins=min(50, len(set(self.sentences_data))), 
                       alpha=0.7, color=colors[0], edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title(f'전체 문장 길이 분포', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(f'문장 길이 ({unit})')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_val = np.mean(self.sentences_data)
        median_val = np.median(self.sentences_data)
        axes[0, 0].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'평균: {mean_val:.1f}')
        axes[0, 0].axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'중간값: {median_val:.1f}')
        axes[0, 0].legend()
        
        # 2. 문장 길이 박스플롯
        box_plot = axes[0, 1].boxplot(self.sentences_data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor(colors[1])
        box_plot['boxes'][0].set_alpha(0.7)
        axes[0, 1].set_title(f'문장 길이 박스플롯', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel(f'문장 길이 ({unit})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 요청별 문장 수 분포
        if self.requests_data.get('sentence_counts'):
            axes[1, 0].hist(self.requests_data['sentence_counts'], 
                           bins=min(30, len(set(self.requests_data['sentence_counts']))), 
                           alpha=0.7, color=colors[2], edgecolor='black', linewidth=0.5)
            axes[1, 0].set_title('요청별 문장 수 분포', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('요청당 문장 수')
            axes[1, 0].set_ylabel('빈도')
            axes[1, 0].grid(True, alpha=0.3)
            
            avg_count = np.mean(self.requests_data['sentence_counts'])
            axes[1, 0].axvline(avg_count, color='red', linestyle='--', alpha=0.8, 
                              label=f'평균: {avg_count:.1f}')
            axes[1, 0].legend()
        
        # 4. 요청별 평균 문장 길이 분포
        if self.requests_data.get('avg_lengths'):
            axes[1, 1].hist(self.requests_data['avg_lengths'], 
                           bins=min(30, len(set(self.requests_data['avg_lengths']))), 
                           alpha=0.7, color=colors[3], edgecolor='black', linewidth=0.5)
            axes[1, 1].set_title(f'요청별 평균 문장 길이 분포', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel(f'평균 문장 길이 ({unit})')
            axes[1, 1].set_ylabel('빈도')
            axes[1, 1].grid(True, alpha=0.3)
            
            avg_length = np.mean(self.requests_data['avg_lengths'])
            axes[1, 1].axvline(avg_length, color='red', linestyle='--', alpha=0.8, 
                              label=f'평균: {avg_length:.1f}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 차트가 저장되었습니다: {save_path}")
        else:
            plt.show()
        
        self.log("✅ 시각화 완료!")
    
    def generate_report(self, unit='characters', save_path: str = None):
        """분석 보고서 생성"""
        if not self.sentences_data:
            print("분석할 데이터가 없습니다.")
            return
        
        report_lines = []
        
        # 헤더
        header = f"LLM 문장 길이 분석 보고서 ({unit} 기준)"
        report_lines.append("="*70)
        report_lines.append(f"    {header}")
        report_lines.append("="*70)
        report_lines.append(f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 모델 정보
        if self.model and self.current_model_name:
            report_lines.append(f"\n🤖 사용된 모델:")
            report_lines.append(f"  • 모델명: {self.current_model_name}")
            report_lines.append(f"  • 파라미터 수: {self.model.num_parameters():,}")
            report_lines.append(f"  • 모델 타입: {type(self.model).__name__}")
        
        # 기본 통계
        report_lines.append(f"\n📊 기본 통계:")
        report_lines.append(f"  • 총 문장 수: {len(self.sentences_data):,}")
        report_lines.append(f"  • 평균 길이: {np.mean(self.sentences_data):.2f} {unit}")
        report_lines.append(f"  • 중간값: {np.median(self.sentences_data):.2f} {unit}")
        report_lines.append(f"  • 표준편차: {np.std(self.sentences_data):.2f} {unit}")
        report_lines.append(f"  • 최소 길이: {min(self.sentences_data)} {unit}")
        report_lines.append(f"  • 최대 길이: {max(self.sentences_data)} {unit}")
        
        # 분위수 정보
        report_lines.append(f"\n📈 분위수 정보:")
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(self.sentences_data, p)
            report_lines.append(f"  • {p:2d}%: {value:6.1f} {unit}")
        
        # 길이별 분포
        report_lines.append(f"\n📏 길이별 분포:")
        if unit == 'characters':
            ranges = [(0, 10), (11, 30), (31, 50), (51, 100), (101, 200), (201, 500), (501, float('inf'))]
            labels = ['극도로 짧음', '매우 짧음', '짧음', '보통', '김', '매우 김', '극도로 김']
        elif unit == 'tokens':
            ranges = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
            labels = ['극도로 짧음', '매우 짧음', '짧음', '보통', '김', '매우 김']
        else:  # words
            ranges = [(0, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, float('inf'))]
            labels = ['극도로 짧음', '매우 짧음', '짧음', '보통', '김', '매우 김']
        
        for (min_len, max_len), label in zip(ranges, labels):
            if max_len == float('inf'):
                count = sum(1 for x in self.sentences_data if x >= min_len)
                percentage = count/len(self.sentences_data)*100
                report_lines.append(f"  • {label:10s} ({min_len:3d}+ {unit}): {count:5,}개 ({percentage:5.1f}%)")
            else:
                count = sum(1 for x in self.sentences_data if min_len <= x <= max_len)
                percentage = count/len(self.sentences_data)*100
                report_lines.append(f"  • {label:10s} ({min_len:3d}-{max_len:3d} {unit}): {count:5,}개 ({percentage:5.1f}%)")
        
        # 요청 관련 통계
        if self.requests_data.get('sentence_counts'):
            report_lines.append(f"\n📋 요청 관련 통계:")
            report_lines.append(f"  • 총 요청 수: {len(self.requests_data['sentence_counts']):,}")
            report_lines.append(f"  • 요청당 평균 문장 수: {np.mean(self.requests_data['sentence_counts']):.2f}")
            report_lines.append(f"  • 요청당 문장 수 범위: {min(self.requests_data['sentence_counts'])} - {max(self.requests_data['sentence_counts'])}")
            
            if self.requests_data.get('avg_lengths'):
                report_lines.append(f"  • 요청별 평균 길이: {np.mean(self.requests_data['avg_lengths']):.2f} {unit}")
        
        # 보고서 출력
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # 파일 저장
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n💾 보고서가 저장되었습니다: {save_path}")
        
        return report_text
    
    def compare_models(self, model_names: List[str], test_prompts: List[str], 
                      unit='characters', max_length: int = 80):
        """여러 모델 비교 분석"""
        print(f"🔬 모델 비교 분석")
        print("="*50)
        
        results = {}
        
        for i, model_name in enumerate(model_names):
            print(f"\n📊 모델 {i+1}/{len(model_names)}: {model_name}")
            print("-" * 30)
            
            if self.load_local_model(model_name):
                stats, responses = self.analyze_generated_responses(
                    test_prompts,
                    unit=unit,
                    max_length=max_length,
                    show_responses=False
                )
                
                if stats:
                    results[model_name] = {
                        'stats': stats,
                        'responses': responses
                    }
                    print(f"✅ {model_name} 분석 완료")
                else:
                    print(f"{model_name} 분석 실패")
            else:
                print(f"{model_name} 로드 실패")
            
            # 메모리 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                del self.tokenizer
                del self.pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 비교 결과 출력
        if results:
            print(f"\n📊 모델 비교 결과")
            print("="*50)
            
            comparison_data = []
            for model_name, result in results.items():
                stats = result['stats']
                comparison_data.append({
                    '모델': model_name,
                    f'평균 길이 ({unit})': f"{stats['avg_sentence_length']:.2f}",
                    '총 문장 수': stats['total_sentences'],
                    '요청당 문장 수': f"{stats['avg_sentences_per_request']:.2f}",
                    '표준편차': f"{stats['std_sentence_length']:.2f}"
                })
            
            # 테이블 형태로 출력
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
        
        return results


def create_argument_parser():
    """명령행 인수 파서 생성"""
    parser = argparse.ArgumentParser(
        description="로컬 LLM 문장 길이 분석기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 프롬프트로 분석
  python %(prog)s --model microsoft/DialoGPT-small --prompts "안녕하세요" "파이썬이란?"
  
  # 데이터셋 분석
  python %(prog)s --dataset OpenAssistant/oasst1 --unit words --sample-size 500
  
  # 모델 비교
  python %(prog)s --compare microsoft/DialoGPT-small google/flan-t5-small --prompts "Hello" "AI란?"
  
  # 파일에서 프롬프트 읽기
  python %(prog)s --model skt/kogpt2-base-v2 --prompts-file prompts.txt --output results/
        """
    )
    
    # 기본 옵션
    parser.add_argument('--model', '-m', type=str, 
                       help='사용할 모델명 (예: microsoft/DialoGPT-small)')
    
    parser.add_argument('--dataset', '-d', type=str,
                       help='분석할 데이터셋명 (예: OpenAssistant/oasst1)')
    
    parser.add_argument('--prompts', '-p', nargs='+', type=str,
                       help='분석할 프롬프트들')
    
    parser.add_argument('--prompts-file', type=str,
                       help='프롬프트가 저장된 파일 경로 (한 줄에 하나씩)')
    
    # 분석 설정
    parser.add_argument('--unit', '-u', type=str, choices=['characters', 'words', 'tokens'], 
                       default='characters', help='측정 단위 (기본값: characters)')
    
    parser.add_argument('--max-length', type=int, default=100,
                       help='최대 생성 길이 (기본값: 100)')
    
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='Temperature 값 (기본값: 0.7)')
    
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='데이터셋 샘플 크기 (기본값: 1000)')
    
    parser.add_argument('--language', type=str,
                       help='언어 필터 (예: ko, en)')
    
    parser.add_argument('--text-field', type=str,
                       help='데이터셋에서 사용할 텍스트 필드명')
    
    # 출력 설정
    parser.add_argument('--output', '-o', type=str,
                       help='출력 디렉토리 경로')
    
    parser.add_argument('--no-plot', action='store_true',
                       help='시각화 비활성화')
    
    parser.add_argument('--no-report', action='store_true',
                       help='보고서 비활성화')
    
    parser.add_argument('--show-responses', action='store_true',
                       help='생성된 응답 출력')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='진행 상황 출력 비활성화')
    
    # 모델 비교
    parser.add_argument('--compare', nargs='+', type=str,
                       help='비교할 모델들 목록')
    
    # 디바이스 설정
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], 
                       default='auto', help='사용할 디바이스 (기본값: auto)')
    
    # 도움말
    parser.add_argument('--list-models', action='store_true',
                       help='추천 모델 목록 출력')
    
    parser.add_argument('--list-datasets', action='store_true',
                       help='추천 데이터셋 목록 출력')
    
    return parser


def load_prompts_from_file(file_path: str) -> List[str]:
    """파일에서 프롬프트 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    except Exception as e:
        print(f"프롬프트 파일 로딩 실패: {e}")
        return []


def main():
    """메인 실행 함수"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 도움말 옵션
    if args.list_models:
        analyzer = LocalLLMSentenceAnalyzer(verbose=False)
        print("🤖 추천 모델 목록:")
        print("="*50)
        for category, models in analyzer.recommended_models.items():
            print(f"\n{category}:")
            for model in models:
                print(f"  • {model}")
        return
    
    if args.list_datasets:
        print("📊 추천 데이터셋 목록:")
        print("="*50)
        datasets = {
            "대화형 데이터셋": [
                "OpenAssistant/oasst1",
                "facebook/empathetic_dialogues",
                "daily_dialog",
            ],
            "QA 데이터셋": [
                "squad", "squad_v2", 
                "natural_questions",
                "ms_marco",
            ],
            "한국어 데이터셋": [
                "squad_kor_v1",
                "klue",
                "korquad",
            ]
        }
        for category, dataset_list in datasets.items():
            print(f"\n{category}:")
            for dataset in dataset_list:
                print(f"  • {dataset}")
        return
    
    # 분석기 초기화
    analyzer = LocalLLMSentenceAnalyzer(verbose=not args.quiet)
    
    # 출력 디렉토리 설정
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        plot_path = os.path.join(args.output, f"analysis_plot_{int(time.time())}.png")
        report_path = os.path.join(args.output, f"analysis_report_{int(time.time())}.txt")
    else:
        plot_path = None
        report_path = None
    
    # 모델 비교 모드
    if args.compare:
        if not args.prompts and not args.prompts_file:
            print("모델 비교를 위해 --prompts 또는 --prompts-file이 필요합니다.")
            return
        
        # 프롬프트 로드
        if args.prompts_file:
            test_prompts = load_prompts_from_file(args.prompts_file)
        else:
            test_prompts = args.prompts
        
        if not test_prompts:
            print("유효한 프롬프트가 없습니다.")
            return
        
        # 모델 비교 실행
        results = analyzer.compare_models(
            args.compare, 
            test_prompts, 
            unit=args.unit,
            max_length=args.max_length
        )
        
        # 결과 저장
        if args.output and results:
            comparison_path = os.path.join(args.output, f"model_comparison_{int(time.time())}.json")
            with open(comparison_path, 'w', encoding='utf-8') as f:
                # JSON 직렬화 가능한 형태로 변환
                json_results = {}
                for model_name, result in results.items():
                    json_results[model_name] = {
                        'stats': result['stats'],
                        'sample_response': result['responses'][0] if result['responses'] else ""
                    }
                json.dump(json_results, f, ensure_ascii=False, indent=2)
            print(f"💾 비교 결과가 저장되었습니다: {comparison_path}")
        
        return
    
    # 단일 모델 분석 모드
    if args.model:
        # 모델 로드
        if not analyzer.load_local_model(args.model, device=args.device):
            print("모델 로딩에 실패했습니다.")
            return
        
        # 프롬프트 분석
        if args.prompts or args.prompts_file:
            if args.prompts_file:
                prompts = load_prompts_from_file(args.prompts_file)
            else:
                prompts = args.prompts
            
            if not prompts:
                print("유효한 프롬프트가 없습니다.")
                return
            
            stats, responses = analyzer.analyze_generated_responses(
                prompts,
                unit=args.unit,
                max_length=args.max_length,
                temperature=args.temperature,
                show_responses=args.show_responses
            )
            
            if stats:
                # 시각화
                if not args.no_plot:
                    analyzer.plot_distributions(unit=args.unit, save_path=plot_path)
                
                # 보고서
                if not args.no_report:
                    analyzer.generate_report(unit=args.unit, save_path=report_path)
        else:
            print("모델 분석을 위해 --prompts 또는 --prompts-file이 필요합니다.")
    
    # 데이터셋 분석 모드  
    elif args.dataset:
        # 데이터셋 로드
        if not analyzer.load_dataset(
            args.dataset,
            sample_size=args.sample_size,
            language=args.language
        ):
            print("데이터셋 로딩에 실패했습니다.")
            return
        
        # 데이터셋 분석
        stats = analyzer.analyze_existing_data(
            text_field=args.text_field,
            unit=args.unit
        )
        
        if stats:
            # 시각화
            if not args.no_plot:
                analyzer.plot_distributions(unit=args.unit, save_path=plot_path)
            
            # 보고서
            if not args.no_report:
                analyzer.generate_report(unit=args.unit, save_path=report_path)
    
    else:
        print("--model 또는 --dataset 중 하나는 필수입니다.")
        print("도움말: python {} --help".format(sys.argv[0]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()