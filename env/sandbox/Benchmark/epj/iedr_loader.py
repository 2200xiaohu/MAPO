"""
IEDR批量结果加载器

用于从预先计算的iedr_batch_results.json中加载IEDR数据
避免在每次对话运行时重新计算
"""

import json
from pathlib import Path
from typing import Optional, Dict


class IEDRLoader:
    """IEDR批量结果加载器"""
    
    def __init__(
        self,
        results_file: str = "Benchmark/topics/new_data/iedr_batch_results_new.json",
    ):
        """
        初始化加载器
        
        Args:
            results_file: IEDR批量结果文件路径（相对于项目根目录）
        """
        # 找到项目根目录
        current_dir = Path(__file__).resolve()
        # 向上找到Benchmark目录
        while current_dir.name != "Benchmark" and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        
        self.project_root = current_dir
        self.results_path = self.project_root / results_file
        self._data = None
        self._index = None
    
    def load(self):
        """加载IEDR数据"""
        if self._data is None:
            if not self.results_path.exists():
                raise FileNotFoundError(
                    f"IEDR批量结果文件不存在: {self.results_path}\n"
                    f"请先运行 batch_evaluate_iedr.py 生成IEDR结果"
                )
            
            with open(self.results_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            
            # 建立索引：script_id -> 数据
            self._index = {item['script_id']: item for item in self._data}
            
            print(f"✅ 已加载 {len(self._data)} 个剧本的IEDR数据")
        
        return self._data
    
    def get_iedr(self, script_id: str) -> Optional[Dict]:
        """
        获取指定剧本的IEDR数据
        
        Args:
            script_id: 剧本ID，格式如 "script_001" 或 "001"
        
        Returns:
            dict: IEDR数据，格式：
            {
                "script_id": "script_001",
                "status": "success",
                "iedr": {
                    "C.1": 3, "C.2": 2, "C.3": 2,
                    "A.1": 3, "A.2": 2, "A.3": 3,
                    "P.1": 0, "P.2": 3, "P.3": 1,
                    "_detailed": {...}
                },
                "P_0": {
                    "C": -16,
                    "A": -21,
                    "P": -15,
                    "total": -52
                },
                "timestamp": "..."
            }
            
            如果找不到或失败，返回None
        """
        # 确保数据已加载
        if self._data is None:
            self.load()
        
        # 规范化script_id格式
        # 如果传入"001"，转换为"script_001"
        if not script_id.startswith("script_"):
            script_id = f"script_{script_id}"
        
        # 从索引中查找
        item = self._index.get(script_id)
        
        if item is None:
            print(f"⚠️  未找到剧本 {script_id} 的IEDR数据")
            return None
        
        if item['status'] != 'success':
            print(f"⚠️  剧本 {script_id} 的IEDR评估失败: {item.get('error', 'Unknown error')}")
            return None
        
        return item
    
    def get_P_0(self, script_id: str) -> Optional[tuple]:
        """
        获取初始赤字向量P_0
        
        Args:
            script_id: 剧本ID
        
        Returns:
            tuple: (C, A, P) 或 None
        """
        item = self.get_iedr(script_id)
        if item is None:
            return None
        
        P_0_dict = item.get('P_0')
        if P_0_dict is None:
            return None
        
        return (P_0_dict['C'], P_0_dict['A'], P_0_dict['P'])
    
    def get_iedr_dict(self, script_id: str) -> Optional[Dict]:
        """
        获取IEDR量表字典
        
        Args:
            script_id: 剧本ID
        
        Returns:
            dict: IEDR量表 或 None
        """
        item = self.get_iedr(script_id)
        if item is None:
            return None
        
        return item.get('iedr')


# 全局单例
_loader_instance = None


def get_iedr_loader() -> IEDRLoader:
    """获取全局IEDR加载器实例"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = IEDRLoader()
    return _loader_instance


def load_precomputed_iedr(script_id: str) -> Optional[Dict]:
    """
    加载预先计算的IEDR数据（便捷函数）
    
    Args:
        script_id: 剧本ID（如"001"或"script_001"）
    
    Returns:
        dict: 完整的IEDR数据项 或 None
    """
    loader = get_iedr_loader()
    return loader.get_iedr(script_id)


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("测试 IEDR 加载器")
    print("=" * 70)
    
    loader = IEDRLoader()
    
    # 测试加载
    print("\n1. 测试加载数据...")
    loader.load()
    
    # 测试获取IEDR
    print("\n2. 测试获取 script_001 的IEDR...")
    data = loader.get_iedr("001")
    if data:
        print(f"✅ script_id: {data['script_id']}")
        print(f"✅ status: {data['status']}")
        print(f"✅ IEDR: C.1={data['iedr']['C.1']}, C.2={data['iedr']['C.2']}, C.3={data['iedr']['C.3']}")
        print(f"✅ P_0: C={data['P_0']['C']}, A={data['P_0']['A']}, P={data['P_0']['P']}")
        print(f"✅ 总赤字: {data['P_0']['total']}")
    
    # 测试便捷函数
    print("\n3. 测试便捷函数...")
    P_0 = loader.get_P_0("001")
    print(f"✅ P_0 tuple: {P_0}")
    
    iedr_dict = loader.get_iedr_dict("001")
    print(f"✅ IEDR dict keys: {list(iedr_dict.keys())}")
    
    print("\n✅ 测试完成")

