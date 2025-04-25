import os
import requests
import time
import json
from tqdm import tqdm

class GeminiAssessor:
    def __init__(self, api_key):
        import google.generativeai as genai
        self.genai = genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    
    def call(self, prompt, temperature=1.0):
        try:
            response = self.model.generate_content(prompt, temperature=temperature)
            return response.text
        except Exception as e:
            print(f"Gemini API调用失败: {e}")
            return None
    
    def retry_call(self, prompt, temperature=0.2, attempts=3, base_delay=60):
        for attempt in range(attempts):
            try:
                return self.call(prompt, temperature)
            except Exception as e:
                print(f"请求失败（尝试 {attempt + 1}/{attempts}）：", e)
            if attempt < attempts - 1:
                time.sleep(base_delay * (attempt + 1))
        return None
    
    def assess_relevance(self, summary, temperature=0.2):
        """Assess if a paper is relevant to search, advertising, or recommendation systems"""
        prompt = """请评估以下论文摘要是否与搜索系统(Search)、广告技术(Advertising)或推荐系统(Recommendation)相关。
        详细分析论文内容，判断其是否讨论了搜索引擎、信息检索、计算广告、点击率预估、个性化推荐、CTR预测、排序算法等相关技术。
        如果相关，请返回'Yes'并简要说明原因；如果不相关，请返回'No'。
        
        论文摘要:
        """
        full_prompt = prompt + summary
        response = self.retry_call(full_prompt, temperature)
        if response is None:
            return False  # Default to not relevant if API call fails
        
        # Check if response contains "yes"
        return "yes" in response.strip().lower()
    
    def translate_summary(self, summary, temperature=0.7):
        """Translate a paper summary from English to Chinese using Gemini"""
        prompt = """请将以下英文论文摘要翻译成中文，保持学术准确性和专业术语的正确翻译：
        
        """
        full_prompt = prompt + summary
        response = self.retry_call(full_prompt, temperature)
        if response is None:
            return ""  # Return empty string if translation fails
        return response

def filter_relevant_papers(papers, gemini_api_key=None):
    """Filter papers relevant to search, advertising, and recommendation systems"""
    if gemini_api_key is None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", None)
        if gemini_api_key is None:
            raise Exception("未设置GEMINI_API_KEY环境变量")
    
    assessor = GeminiAssessor(api_key=gemini_api_key)
    relevant_papers = []
    
    print('[+] 使用Gemini API筛选搜广推相关论文....')
    
    for paper in tqdm(papers, desc="论文筛选进度"):
        summary = paper['summary']
        if assessor.assess_relevance(summary):
            relevant_papers.append(paper)
    
    print(f'[+] 总论文数: {len(papers)} | 相关论文数: {len(relevant_papers)}')
    
    return relevant_papers