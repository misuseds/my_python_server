import tkinter as tk
from tkinter import filedialog, messagebox
import ezdxf
import os

def select_dxf_files():
    """
    使用弹窗选择多个DXF文件
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框，允许选择多个文件
    file_paths = filedialog.askopenfilenames(
        title="选择要合并的DXF文件",
        filetypes=[
            ("DXF files", "*.dxf"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return list(file_paths)

def is_supported_entity(entity):
    """
    检查实体是否为支持的类型
    """
    unsupported_types = ['DIMENSION', 'LEADER', 'MLINE', 'HATCH']
    return entity.dxftype() not in unsupported_types

def merge_dxf_files(input_files, output_file):
    """
    合并多个DXF文件
    
    Args:
        input_files: 输入的DXF文件路径列表
        output_file: 输出文件路径
    """
    if not input_files:
        raise ValueError("没有选择任何文件")
    
    # 创建新的DXF文档
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    
    # 记录处理状态
    processed_files = []
    errors = []
    
    # 遍历所有输入文件
    for i, file_path in enumerate(input_files):
        try:
            # 读取每个DXF文件
            source_doc = ezdxf.readfile(file_path)
            source_msp = source_doc.modelspace()
            
            # 复制图层定义
            for layer in source_doc.layers:
                if layer.dxf.name not in doc.layers:
                    try:
                        doc.layers.new(
                            layer.dxf.name,
                            dxfattribs={
                                'color': layer.dxf.color,
                                'linetype': layer.dxf.linetype,
                            }
                        )
                    except Exception as e:
                        print(f"警告: 图层 {layer.dxf.name} 复制失败: {e}")
            
            # 复制线型定义（已修复）
            for linetype in source_doc.linetypes:
                if linetype.dxf.name not in doc.linetypes:
                    try:
                        # 安全地处理线型属性
                        linetype_attribs = {
                            'description': getattr(linetype.dxf, 'description', ''),
                        }
                        
                        # 只有在pattern属性存在且有效时才添加
                        if hasattr(linetype.dxf, 'pattern'):
                            pattern = linetype.dxf.pattern
                            # 验证pattern是否为有效的格式
                            if isinstance(pattern, (list, tuple)) and len(pattern) > 0:
                                linetype_attribs['pattern'] = pattern
                        
                        doc.linetypes.new(
                            linetype.dxf.name,
                            dxfattribs=linetype_attribs
                        )
                    except Exception as e:
                        # 如果复制失败，尝试创建不带pattern的线型
                        print(f"警告: 线型 {linetype.dxf.name} 复制失败: {e}")
                        try:
                            doc.linetypes.new(linetype.dxf.name)
                        except:
                            pass  # 如果仍然失败，跳过该线型
            
            # 复制文字样式定义
            for style in source_doc.styles:
                if style.dxf.name not in doc.styles:
                    try:
                        doc.styles.new(
                            style.dxf.name,
                            dxfattribs={
                                'font': getattr(style.dxf, 'font', 'Arial'),
                            }
                        )
                    except Exception as e:
                        print(f"警告: 文字样式 {style.dxf.name} 复制失败: {e}")
            
            # 复制块定义
            for block in source_doc.blocks:
                if block.name not in doc.blocks:
                    try:
                        new_block = doc.blocks.new(block.name)
                        for entity in block:
                            try:
                                # 只处理支持的实体类型
                                if is_supported_entity(entity):
                                    new_block.add_foreign_entity(entity, source_doc)
                            except Exception as e:
                                print(f"警告: 块 {block.name} 中实体复制失败: {e}")
                    except Exception as e:
                        print(f"警告: 块 {block.name} 复制失败: {e}")
            
            # 复制实体到目标文件
            entity_count = 0
            for entity in source_msp:
                try:
                    if entity.dxftype() == 'INSERT':
                        # 特殊处理INSERT实体
                        block_name = entity.dxf.name
                        if block_name in source_doc.blocks:
                            # 确保块定义已经复制
                            if block_name not in doc.blocks:
                                source_block = source_doc.blocks[block_name]
                                try:
                                    new_block = doc.blocks.new(block_name)
                                    for block_entity in source_block:
                                        try:
                                            # 只处理支持的实体类型
                                            if is_supported_entity(block_entity):
                                                new_block.add_foreign_entity(block_entity, source_doc)
                                        except Exception as e:
                                            print(f"警告: 块 {block_name} 中实体复制失败: {e}")
                                except Exception as e:
                                    print(f"警告: 块 {block_name} 创建失败: {e}")
                            
                            # 添加INSERT实体
                            try:
                                msp.add_entity(entity.copy())
                                entity_count += 1
                            except Exception as e:
                                print(f"警告: INSERT实体 {block_name} 复制失败: {e}")
                    else:
                        # 处理其他类型的实体，但跳过不支持的类型
                        if is_supported_entity(entity):
                            msp.add_foreign_entity(entity, source_doc)
                            entity_count += 1
                        else:
                            print(f"跳过不支持的实体类型: {entity.dxftype()}")
                except Exception as e:
                    print(f"警告: 实体复制失败: {e}")
            
            processed_files.append(f"{os.path.basename(file_path)} ({entity_count}个实体)")
            
        except Exception as e:
            error_msg = f"处理文件 {file_path} 时出错: {str(e)}"
            errors.append(error_msg)
            print(error_msg)
    
    # 保存合并后的文件
    doc.saveas(output_file)
    
    return processed_files, errors

def main():
    """
    主函数：图形界面合并DXF文件，保存在第一个文件的目录下
    """
    try:
        # 选择多个DXF文件
        print("请选择要合并的DXF文件...")
        input_files = select_dxf_files()
        
        if not input_files:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("提示", "未选择任何文件")
            root.destroy()
            return
        
        print(f"已选择 {len(input_files)} 个文件:")
        for i, file in enumerate(input_files, 1):
            print(f"  {i}. {file}")
        
        # 确定输出文件路径（在第一个文件的同一目录下）
        first_file_dir = os.path.dirname(input_files[0])
        output_file = os.path.join(first_file_dir, "merged_dxf_files.dxf")
        
        print(f"\n输出文件将保存到: {output_file}")
        
        # 执行合并操作
        print("\n正在合并文件...")
        processed_files, errors = merge_dxf_files(input_files, output_file)
        
        # 显示结果
        result_message = f"成功合并 {len(processed_files)} 个文件到:\n{output_file}\n\n"
        
        if processed_files:
            result_message += "已处理的文件:\n"
            for file_info in processed_files:
                result_message += f"  • {file_info}\n"
        
        if errors:
            result_message += f"\n错误信息 ({len(errors)} 个):\n"
            for error in errors[:5]:  # 只显示前5个错误
                result_message += f"  • {error}\n"
            if len(errors) > 5:
                result_message += f"  ... 还有 {len(errors) - 5} 个错误\n"
        
        # 显示结果对话框
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("合并完成", result_message)
        root.destroy()
        
        print(result_message)
        
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        print(error_msg)
        
        # 显示错误对话框
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", error_msg)
        root.destroy()

if __name__ == "__main__":
    main()