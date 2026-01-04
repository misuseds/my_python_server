import win32com.client
import pythoncom

def get_first_body_bounding_box_and_add_spec():
    """
    获取SolidWorks当前零件的第一个实体的边界框尺寸，并添加到文件属性中
    """
    try:
        # 初始化COM库
        pythoncom.CoInitialize()

        # 连接到SolidWorks应用程序
        sw_app = win32com.client.Dispatch("SldWorks.Application")
        
        # 获取活动文档
        active_doc = sw_app.ActiveDoc
        if not active_doc:
            print("没有打开的文档")
            return False
            
        # 检查是否为零件文档 (1表示零件文档)
        if active_doc.GetType != 1:  
            print("当前文档不是零件文档")
            return False
        
        # 获取零件文档对象
        part_doc = active_doc
        
        # 获取模型的实体集合
        bodies = part_doc.GetBodies2(0, False)  # 0 表示实体 (Solid Bodies)，False表示获取所有实体
        
        if not bodies or len(bodies) == 0:
            print("未找到任何实体")
            return False
        
        # 获取第一个实体
        first_body = bodies[0]
        
        # 获取该实体的边界框

        
        bbox = first_body.GetBodyBox()  # 使用替代方法GetBodyBox
        
        # 确保正确解包边界框值
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 6:
            xmin, ymin, zmin, xmax, ymax, zmax = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
        else:
            print("无法获取正确的边界框数据")
            return False
        
        # 计算长宽高（单位通常为米，需要转换为毫米）
        length = (xmax - xmin) * 1000  # 转换为毫米
        width = (ymax - ymin) * 1000   # 转换为毫米
        height = (zmax - zmin) * 1000  # 转换为毫米
        dimensions = sorted([length, width, height])
        height  = dimensions[0]
        length= dimensions[1]
        width= dimensions[2]
        
        # 创建规格字符串: PL厚度x长x宽
        spec_string = f"PL{height:.1f}x{length:.1f}x{width:.1f}"
        print(f"第一个实体边界框信息 - 长: {length:.2f}mm, 宽: {width:.2f}mm, 厚: {height:.2f}mm")        
        
        # 获取默认配置名称
        config_names = part_doc.GetConfigurationNames
        config_name = config_names[0] if config_names else ""
        
        # 使用默认配置的自定义属性管理器
        custom_prop_mgr = part_doc.Extension.CustomPropertyManager("")
        
        # 先尝试设置现有属性，如果不存在则添加新属性
        result = custom_prop_mgr.Set("规格", spec_string)
        if not result:
            # 如果设置失败，说明属性不存在，需要添加
            custom_prop_mgr.Add3("规格", 30, spec_string, 1)  # 30对应swCustomInfoText，1对应swCustomPropertyReplaceValue
        
        print(f"成功添加规格属性: {spec_string}")
        
        return True
        
    except Exception as e:
        print(f"操作失败: {str(e)}")
        return False
    finally:
        # 清理COM资源
        pythoncom.CoUninitialize()

# 主函数调用示例
if __name__ == "__main__":
    get_first_body_bounding_box_and_add_spec()