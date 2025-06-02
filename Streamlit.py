import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 设置页面配置
st.set_page_config(page_title="图片分析系统", layout="wide", page_icon="🔍")

# 模拟用户数据库（仅用于演示，实际应用中应使用加密存储）
USER_DB = {"admin": "password123"}

# 初始化会话状态
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# 加载 YOLO 模型（使用缓存避免重复加载）
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # 替换为您的模型路径

model = load_model()

# 登录页面
def login_page():
    st.title("用户登录")
    st.write("请输入用户名和密码以进入图片分析系统")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        if username in USER_DB and USER_DB[username] == password:
            st.session_state["logged_in"] = True
            st.success("登录成功！即将进入图片分析界面...")
            st.rerun()
        else:
            st.error("用户名或密码错误，请重试")

# 辅助函数：获取类别和颜色
def get_category_and_color(label):
    label_lower = label.lower()
    if "unripe" in label_lower:
        return "unripe", "#FFC107"
    elif "rotten" in label_lower:
        return "rotten", "#F44336"
    else:
        return "fresh", "#8BC34A"

# 图片分析页面
def analysis_page():
    st.title("图片分析系统")
    st.write("欢迎使用！请上传图片并设置参数以进行目标检测")

    # 图片上传
    uploaded_file = st.file_uploader("选择一张图片", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.info("图片预览")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("检测设置")
            confidence_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25)
            button = st.button("开始检测")

            if button:
                with st.spinner("正在检测目标，请稍候..."):
                    # 运行 YOLO 模型
                    results = model(image, conf=confidence_threshold)
                    result_image = Image.fromarray(results[0].plot())

                    # 获取检测结果
                    detections = [
                        {"label": model.names[int(box.cls.item())], "confidence": float(box.conf.item())}
                        for box in results[0].boxes
                    ]

                    # 显示检测结果图像
                    st.subheader("检测结果")
                    st.caption("基于 YOLOv8 的目标检测")
                    st.image(result_image, use_container_width=True)

                    # 显示检测对象列表
                    st.subheader("检测对象列表")
                    for i, detection in enumerate(detections):
                        category, color = get_category_and_color(detection["label"])
                        st.write(
                            f"{i+1}. **{detection['label']}** - 置信度: {detection['confidence']:.2f}",
                            f"<span style='background-color: {color}; width: 20px; height: 20px; display: inline-block;'></span>",
                            unsafe_allow_html=True
                        )

                    # 计算并显示统计信息
                    fresh_count = sum(1 for d in detections if "fresh" in get_category_and_color(d["label"])[0])
                    unripe_count = sum(1 for d in detections if "unripe" in get_category_and_color(d["label"])[0])
                    rotten_count = sum(1 for d in detections if "rotten" in get_category_and_color(d["label"])[0])
                    total = len(detections) or 1  # 防止除零
                    st.subheader("分析总结")
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.metric("新鲜水果", f"{(fresh_count / total * 100):.1f}%")
                    with col4:
                        st.metric("未成熟水果", f"{(unripe_count / total * 100):.1f}%")
                    with col5:
                        st.metric("腐烂水果", f"{(rotten_count / total * 100):.1f}%")

# 主逻辑
if not st.session_state["logged_in"]:
    login_page()
else:
    analysis_page()
