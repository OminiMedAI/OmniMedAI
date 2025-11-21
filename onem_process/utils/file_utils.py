"""
文件工具函数
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
import logging


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files_by_extension(directory: Union[str, Path], 
                           extensions: List[str],
                           recursive: bool = True) -> List[Path]:
    """根据扩展名查找文件"""
    directory = Path(directory)
    files = []
    
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        files.extend(directory.glob(f"{pattern}{ext}"))
        files.extend(directory.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(list(set(files)))


def find_files_by_pattern(directory: Union[str, Path], 
                         pattern: str,
                         recursive: bool = True) -> List[Path]:
    """根据模式查找文件"""
    directory = Path(directory)
    
    if recursive:
        return sorted(list(directory.rglob(pattern)))
    else:
        return sorted(list(directory.glob(pattern)))


def get_file_hash(file_path: Union[str, Path], 
                 algorithm: str = 'md5',
                 chunk_size: int = 8192) -> str:
    """计算文件哈希值"""
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def copy_file_with_metadata(src: Union[str, Path], 
                           dst: Union[str, Path],
                           preserve_metadata: bool = True) -> Path:
    """复制文件并可选择保留元数据"""
    src = Path(src)
    dst = Path(dst)
    
    ensure_dir(dst.parent)
    
    if preserve_metadata:
        shutil.copy2(src, dst)
    else:
        shutil.copy(src, dst)
    
    return dst


def move_file(src: Union[str, Path], 
              dst: Union[str, Path]) -> Path:
    """移动文件"""
    src = Path(src)
    dst = Path(dst)
    
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))
    
    return dst


def safe_filename(filename: str, replacement_char: str = '_') -> str:
    """生成安全的文件名"""
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in invalid_chars:
        safe_name = safe_name.replace(char, replacement_char)
    
    # 移除控制字符
    safe_name = ''.join(c for c in safe_name if ord(c) >= 32)
    
    # 限制长度
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:255-len(ext)] + ext
    
    return safe_name


def get_unique_filename(file_path: Union[str, Path]) -> Path:
    """获取唯一的文件名（避免重复）"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return file_path
    
    counter = 1
    stem = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent
    
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        
        if not new_path.exists():
            return new_path
        
        counter += 1


def save_json(data: Dict[Any, Any], 
              file_path: Union[str, Path],
              indent: int = 2) -> None:
    """保存数据为 JSON 文件"""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Dict[Any, Any]:
    """加载 JSON 文件"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_directory_size(directory: Union[str, Path]) -> Dict[str, int]:
    """获取目录大小信息"""
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return {'total_size': 0, 'file_count': 0}
    
    total_size = 0
    file_count = 0
    
    for item in directory.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1
    
    return {
        'total_size': total_size,
        'file_count': file_count,
        'total_size_mb': round(total_size / (1024 * 1024), 2)
    }


def clean_directory(directory: Union[str, Path], 
                  keep_patterns: Optional[List[str]] = None) -> None:
    """清理目录中的文件"""
    directory = Path(directory)
    
    if not directory.exists():
        return
    
    for item in directory.iterdir():
        if item.is_file():
            should_keep = False
            
            if keep_patterns:
                for pattern in keep_patterns:
                    if item.match(pattern):
                        should_keep = True
                        break
            
            if not should_keep:
                item.unlink()


def backup_file(file_path: Union[str, Path], 
               backup_dir: Optional[Union[str, Path]] = None,
               suffix: str = '.backup') -> Path:
    """备份文件"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File to backup not found: {file_path}")
    
    # 确定备份目录
    if backup_dir is None:
        backup_dir = file_path.parent / "backups"
    
    backup_dir = Path(backup_dir)
    ensure_dir(backup_dir)
    
    # 生成备份文件名
    backup_name = file_path.name + suffix
    backup_path = backup_dir / backup_name
    
    # 如果备份文件已存在，添加时间戳
    if backup_path.exists():
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}{suffix}"
        backup_path = backup_dir / backup_name
    
    # 复制文件
    copy_file_with_metadata(file_path, backup_path)
    
    return backup_path


def create_file_listing(directory: Union[str, Path], 
                       output_file: Union[str, Path],
                       include_hash: bool = True,
                       recursive: bool = True) -> None:
    """创建文件列表"""
    directory = Path(directory)
    output_file = Path(output_file)
    
    ensure_dir(output_file.parent)
    
    files_info = []
    
    pattern = "**/*" if recursive else "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            info = {
                'path': str(file_path.relative_to(directory)),
                'size': file_path.stat().st_size,
                'modified': file_path.stat().st_mtime
            }
            
            if include_hash:
                try:
                    info['hash'] = get_file_hash(file_path)
                except Exception as e:
                    info['hash'] = f"Error: {e}"
            
            files_info.append(info)
    
    save_json(files_info, output_file)


def validate_file_structure(directory: Union[str, Path],
                          expected_structure: Dict[str, Any]) -> Dict[str, bool]:
    """验证文件结构"""
    directory = Path(directory)
    validation_results = {}
    
    def _check_structure(base_path: Path, structure: Dict[str, Any]) -> None:
        for name, expected in structure.items():
            current_path = base_path / name
            
            if isinstance(expected, dict):
                if not current_path.exists() or not current_path.is_dir():
                    validation_results[name] = False
                else:
                    validation_results[name] = True
                    _check_structure(current_path, expected)
            elif isinstance(expected, list):
                if not current_path.exists() or not current_path.is_dir():
                    validation_results[name] = False
                else:
                    files_found = [f.name for f in current_path.iterdir() if f.is_file()]
                    validation_results[name] = all(req_file in files_found for req_file in expected)
            else:
                validation_results[name] = current_path.exists() and current_path.is_file()
    
    _check_structure(directory, expected_structure)
    return validation_results