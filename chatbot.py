import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# LangChain 和 LangGraph 相关导入
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# 配置环境
import dotenv
dotenv.load_dotenv()  # 加载.env文件中的环境变量

# 实例化API
app = FastAPI(title="简历助手API", description="提供简历创建、职业规划和简历咨询服务")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化LLM模型
llm = ChatOpenAI(
    temperature=0.7, 
    model="gpt-3.5-turbo"  # 根据需要更换为适合的模型
)

# 定义聊天状态模型
class ChatState(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    resume_data: Optional[Dict[str, Any]] = None
    current_mode: str = "general"  # general, resume_building, career_planning, resume_consulting

# 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = None  # 可选，用于切换模式

class ChatResponse(BaseModel):
    response: str
    mode: str

# 系统提示模板
SYSTEM_PROMPTS = {
    "general": """你是一个专业的简历和职业发展顾问。你可以帮助用户从零开始创建简历，提供职业规划建议，
    以及作为简历顾问提供详细的反馈和改进建议。请询问用户想要什么类型的帮助。""",
    
    "resume_building": """作为简历创建助手，你的任务是帮助用户从零开始创建一份专业的简历。
    遵循以下步骤:
    1. 收集个人信息（姓名、联系方式等）
    2. 获取教育背景
    3. 了解工作经历
    4. 询问技能和专长
    5. 了解项目经验
    6. 收集证书和资格认证
    7. 帮助用户组织和格式化简历内容
    
    请一步一步引导用户，确保收集全面而相关的信息。""",
    
    "career_planning": """作为职业规划顾问，你需要帮助用户明确职业目标并制定职业发展路径。
    请考虑以下方面:
    1. 用户的技能、经验和教育背景
    2. 用户的兴趣和激情所在
    3. 行业趋势和就业前景
    4. 短期和长期职业目标
    5. 可能的职业发展路径
    6. 为实现这些目标所需的技能和资格
    
    提供具体、可行的建议，并鼓励用户设定明确的职业发展步骤。""",
    
    "resume_consulting": """作为简历顾问，你需要分析用户的简历并提供详细的反馈。
    评估以下方面:
    1. 简历的结构和格式
    2. 内容的相关性和有效性
    3. 措辞和表达方式
    4. 成就和成果的展示
    5. 针对目标职位的定制程度
    6. ATS友好性
    
    提供具体的改进建议，并解释为什么这些变更会使简历更有效。"""
}

# 解析简历文件
def parse_resume(file_path: str, file_type: str) -> str:
    """解析上传的简历文件并返回文本内容"""
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        else:
            return "不支持的文件类型"
    except Exception as e:
        return f"解析文件时发生错误: {str(e)}"

# 定义状态转换函数
def get_next_step(state: Dict) -> str:
    """决定下一步操作"""
    current_mode = state["current_mode"]
    messages = state["messages"]
    
    # 简单的模式切换逻辑
    if len(messages) >= 2:  # 至少有一轮对话
        last_user_msg = messages[-2]["content"].lower() if messages[-2]["role"] == "human" else ""
        
        # 检测用户是否请求模式切换
        if "创建简历" in last_user_msg or "写简历" in last_user_msg:
            return "switch_to_resume_building"
        elif "职业规划" in last_user_msg or "职业建议" in last_user_msg:
            return "switch_to_career_planning"
        elif "评估简历" in last_user_msg or "简历反馈" in last_user_msg:
            return "switch_to_resume_consulting"
    
    # 默认继续当前模式
    return "continue_conversation"

# 定义不同的处理节点
def switch_to_resume_building(state: Dict) -> Dict:
    """切换到简历创建模式"""
    state["current_mode"] = "resume_building"
    return state

def switch_to_career_planning(state: Dict) -> Dict:
    """切换到职业规划模式"""
    state["current_mode"] = "career_planning"
    return state

def switch_to_resume_consulting(state: Dict) -> Dict:
    """切换到简历咨询模式"""
    state["current_mode"] = "resume_consulting"
    return state

def continue_conversation(state: Dict) -> Dict:
    """继续当前对话模式"""
    messages = state["messages"]
    current_mode = state["current_mode"]
    
    # 准备消息历史
    chat_messages = []
    
    # 添加系统提示
    chat_messages.append(SystemMessage(content=SYSTEM_PROMPTS[current_mode]))
    
    # 添加对话历史
    for msg in messages[:-1]:  # 不包括最新的AI消息
        if msg["role"] == "human":
            chat_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            chat_messages.append(AIMessage(content=msg["content"]))
    
    # 获取LLM响应
    response = llm.invoke(chat_messages)
    
    # 更新状态
    state["messages"].append({
        "role": "ai",
        "content": response.content
    })
    
    return state

# 创建LangGraph
def create_chat_graph():
    """创建聊天流程图"""
    workflow = StateGraph(state_type=Dict)
    
    # 添加节点
    workflow.add_node("switch_to_resume_building", switch_to_resume_building)
    workflow.add_node("switch_to_career_planning", switch_to_career_planning)
    workflow.add_node("switch_to_resume_consulting", switch_to_resume_consulting)
    workflow.add_node("continue_conversation", continue_conversation)
    
    # 添加边和条件
    workflow.add_conditional_edges(
        "root",
        get_next_step,
        {
            "switch_to_resume_building": "switch_to_resume_building",
            "switch_to_career_planning": "switch_to_career_planning",
            "switch_to_resume_consulting": "switch_to_resume_consulting",
            "continue_conversation": "continue_conversation"
        }
    )
    
    # 设置所有节点返回到END
    workflow.add_edge("switch_to_resume_building", END)
    workflow.add_edge("switch_to_career_planning", END)
    workflow.add_edge("switch_to_resume_consulting", END)
    workflow.add_edge("continue_conversation", END)
    
    # 编译
    return workflow.compile()

# 初始化图
chat_graph = create_chat_graph()

# 会话管理
sessions = {}  # 简单的会话存储

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户聊天请求"""
    session_id = "default"  # 简单起见使用默认会话，实际应用中应该使用用户特定的会话ID
    
    # 获取或创建会话状态
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "resume_data": None,
            "current_mode": "general"
        }
    
    # 更新状态
    state = sessions[session_id]
    
    # 如果请求中指定了模式，则切换模式
    if request.mode:
        state["current_mode"] = request.mode
    
    # 添加用户消息
    state["messages"].append({
        "role": "human",
        "content": request.message
    })
    
    # 运行图
    result = chat_graph.invoke(state)
    
    # 更新会话状态
    sessions[session_id] = result
    
    # 获取最新的AI回复
    ai_message = result["messages"][-1]["content"] if result["messages"] and result["messages"][-1]["role"] == "ai" else "抱歉，处理您的请求时出现了问题。"
    
    return ChatResponse(
        response=ai_message,
        mode=result["current_mode"]
    )

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """上传简历文件进行分析"""
    try:
        # 保存上传的文件
        file_path = f"./temp/{file.filename}"
        os.makedirs("./temp", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # 确定文件类型
        file_extension = file.filename.split(".")[-1].lower()
        
        # 解析简历
        resume_text = parse_resume(file_path, file_extension)
        
        # 分析简历内容
        prompt = ChatPromptTemplate.from_template(
            """你是一个专业的简历分析专家。请分析以下简历内容，并提取关键信息:
            
            简历内容:
            {resume_content}
            
            请提供以下分析:
            1. 简历的整体结构和格式评估
            2. 内容的优势和不足
            3. 针对性改进建议
            4. ATS友好度评估
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke({"resume_content": resume_text})
        
        # 删除临时文件
        os.remove(file_path)
        
        return JSONResponse(content={"analysis": analysis})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理简历时出错: {str(e)}")

@app.post("/generate-resume")
async def generate_resume(
    personal_info: str = Form(...),
    education: str = Form(...),
    experience: str = Form(...),
    skills: str = Form(...),
    target_position: str = Form(...)
):
    """根据提供的信息生成简历"""
    try:
        prompt = ChatPromptTemplate.from_template(
            """你是一个专业的简历创建专家。根据以下信息，创建一份专业、有吸引力的简历:
            
            个人信息: {personal_info}
            教育背景: {education}
            工作经验: {experience}
            技能: {skills}
            目标职位: {target_position}
            
            请生成一份完整的简历，包括所有必要的部分，格式清晰，重点突出与目标职位相关的经验和技能。
            简历应该是ATS友好的，并遵循现代简历的最佳实践。
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        resume = chain.invoke({
            "personal_info": personal_info,
            "education": education,
            "experience": experience,
            "skills": skills,
            "target_position": target_position
        })
        
        return JSONResponse(content={"resume": resume})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成简历时出错: {str(e)}")

@app.post("/career-advice")
async def get_career_advice(
    current_position: str = Form(...),
    skills: str = Form(...),
    interests: str = Form(...),
    goals: str = Form(...)
):
    """提供个性化职业建议"""
    try:
        prompt = ChatPromptTemplate.from_template(
            """你是一个经验丰富的职业顾问。请根据以下信息提供详细的职业规划建议:
            
            当前职位: {current_position}
            技能: {skills}
            兴趣: {interests}
            职业目标: {goals}
            
            请提供以下建议:
            1. 基于当前技能和兴趣的可能职业路径
            2. 实现职业目标所需的额外技能或资格
            3. 短期（1-2年）职业发展建议
            4. 长期（3-5年）职业规划
            5. 行业趋势和可能的机会
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        advice = chain.invoke({
            "current_position": current_position,
            "skills": skills,
            "interests": interests,
            "goals": goals
        })
        
        return JSONResponse(content={"advice": advice})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取职业建议时出错: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)