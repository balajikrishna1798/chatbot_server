from http.client import HTTPException
from fastapi import FastAPI, HTTPException, Depends, UploadFile
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
import os
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV Variables
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/contus_chatbot")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize API keys and environment variables
Groq_api_key = os.getenv("GROQ_API_KEY")

# Define LLMs and embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

session_store = {}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class QuestionRequest(BaseModel):
    session_id: str
    question: str

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    chunk_ids = Column(String)  # Store chunk IDs as a JSON string
    owner = relationship("User")


class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(String, nullable=False)
    answer_text = Column(String, nullable=True)
    session_id = Column(String, nullable=False)  # Add this line
    owner_id = Column(Integer, ForeignKey("users.id"))  # User who asked the question
    owner = relationship("User")


Base.metadata.create_all(bind=engine)

# Helper functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authentication endpoints
@app.post("/signup", response_model=dict)
def signup(username: str, password: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/login", response_model=dict)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}



# Helper to get session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

@app.post("/upload_files")
async def upload_files(
    files: list[UploadFile],
    current_user: User = Depends(get_current_user),  # Requires authentication
    db: Session = Depends(get_db),
):
    owner_id = current_user.id  # Use the ID of the authenticated user

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")

        # Save file locally
        file_path = f"./uploaded_files/{file.filename}"
        os.makedirs("./uploaded_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Save file metadata in DB
        new_file = UploadedFile(file_name=file.filename, owner_id=owner_id)
        db.add(new_file)
        db.commit()
        db.refresh(new_file)

        # Load and process PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Add metadata and split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)

        vectors = []  # Collect vectors for Pinecone
        chunk_ids = []  # Collect chunk IDs for database
        for idx, chunk in enumerate(splits):
            # Set vector ID dynamically using the filename as a prefix
            vector_id = f"{file.filename}_chunk_{idx + 1}"  # Example: file123_chunk_1
            chunk_ids.append(vector_id)  # Save the vector ID for later reference

            # Prepare the vector for Pinecone
            vector = {
                "id": vector_id,  # Set the vector ID
                "values": embeddings.embed_query(chunk.page_content),  # Embed the chunk's text
                "metadata": {  # Metadata (optional, only if needed)
                "text": chunk.page_content,
                    "type": "pdf",
                    "owner_id": owner_id,
                    "file_id": file.filename,
                },
            }
            vectors.append(vector)

        # Upsert vectors to Pinecone
        vectorDB = PineconeVectorStore(index_name="test-rag",embedding=embeddings)
        vectorDB._index.upsert(vectors=vectors, namespace=f"user_{owner_id}")

        # Save chunk IDs in database
        new_file.chunk_ids = chunk_ids
        db.commit()

    return {"message": "Files uploaded successfully.", "uploaded_files": [file.filename for file in files]}

@app.post("/ask_question")
async def ask_question(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user),  # Requires authentication
    db: Session = Depends(get_db),

):
    session_id = request.session_id
    question = request.question
    user_id = current_user.id  # Authenticated user's ID

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Dynamically load Chroma DB
    vectorDB = PineconeVectorStore(
        index_name="test-rag",
        embedding=embeddings)
    
    retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs={
    "k": 5,
    'filter': {
        'type': "pdf",
        'owner_id': user_id,
    },
    "namespace":f"user_{user_id}"

})

    # Create history-aware retriever
    contextualize_q_system_prompt = """Your task is to transform a user's question into a self-contained query by considering the recent chat history. 
    If the user's question relies on previous context, rewrite it to include all necessary details for it to stand alone. 
    If no reformulation is needed, return the question unchanged. Do not answer the question."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history", n_messages=2),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    # Define QA chain
    qa_system_prompt = """You are a highly accurate assistant for answering user questions. 
    You will use the following retrieved information to provide precise and factual answers. 
    If the retrieved context does not fully address the question or you are unsure of the answer, clearly state, 'I don't know.' 
    Avoid guessing or fabricating responses. Always ensure the answer is concise and free of errors.

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history", n_messages=2),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create conversational RAG chain with history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Invoke the chain
    result = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    
    answer = result["answer"]

    # Save the question and answer in the database
    new_question = Question(question_text=question, answer_text=answer, owner_id=user_id,session_id=session_id)
    db.add(new_question)
    db.commit()

    return {"question": question, "answer": answer}


@app.get("/get_uploaded_files")
async def get_uploaded_files(
    current_user: User = Depends(get_current_user),  # Authenticated user
    db: Session = Depends(get_db),
):
    # Fetch files owned by the current user
    owner_id = current_user.id
    uploaded_files = db.query(UploadedFile).filter(UploadedFile.owner_id == owner_id).all()
    
    # If you store files in a directory, validate ownership before including
    files = [file.file_name for file in uploaded_files]
    return {"uploaded_files": files}

@app.get("/gethistory")
async def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Retrieve the entire chat history for the authenticated user, grouped by session ID.
    """
    # Fetch all questions and answers for the current user
    questions = db.query(Question).filter(Question.owner_id == current_user.id).all()

    if not questions:
        # Return an empty history if no questions exist
        return {"history": []}

    # Group questions and answers by session_id
    sessions = {}
    for question in questions:
        session_id = question.session_id  # Use session_id to group
        if session_id not in sessions:
            sessions[session_id] = []

        sessions[session_id].append({
            "question": question.question_text,
            "answer": question.answer_text,
            "timestamp": question.timestamp if hasattr(question, 'timestamp') else None  # Handle missing timestamps
        })

    # Convert sessions into a list of dicts for easy frontend rendering
    history_data = []
    for session_id, session_data in sessions.items():
        history_data.append({
            "session_id": session_id,
            "messages": session_data
        })

    return {"history": history_data}

@app.delete("/delete_file")
async def delete_file(file_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Delete a PDF file, its metadata, and all associated embeddings from Pinecone for the authenticated user.
    """
    owner_id = current_user.id

    # Query the file in the database
    file = db.query(UploadedFile).filter(UploadedFile.file_name == file_id, UploadedFile.owner_id == owner_id).first()

    if not file:
        raise HTTPException(status_code=404, detail="File not found or not owned by the user")

    try:
        # Initialize Pinecone index
        vectorDB = PineconeVectorStore(embedding=embeddings, index_name="test-rag")
        response_generator = vectorDB._index.list(prefix="Balaji_M_",namespace=f"user_{owner_id}")

        # Collect IDs from the generator
        ids = list(response_generator)  # Iterate through the generator

        if not ids:
            return {"message": "No chunk IDs found in Pinecone for this namespace", "namespace": f"user_{owner_id}"}

        # Print all chunk IDs
        print("Chunk IDs in Pinecone:")
        vectorDB._index.delete(ids=ids,namespace=f"user_{owner_id}")
        # Delete the file from the database
        db.delete(file)
        db.commit()

        # Optionally, delete the file from local storage (if applicable)
        file_path = f"./uploaded_files/{file_id}"
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error during deletion: {str(e)}")

    return {"message": f"File '{file_id}' and its related data have been deleted successfully."}

@app.get("/pinecone_chunk_ids/{namespace}")
async def get_pinecone_chunk_ids(
    namespace: str,
    current_user: User = Depends(get_current_user),  # Authenticated user
):
    """
    Retrieve and print all chunk IDs from Pinecone for a given namespace (e.g., user-specific or file-specific).
    """
    try:
        # Initialize Pinecone index
        vectorDB = PineconeVectorStore(embedding=embeddings, index_name="test-rag")

        # List all vector IDs in the specified namespace
        response_generator = vectorDB._index.list(prefix="Balaji_M_",namespace=namespace)

        # Collect IDs from the generator
        ids = list(response_generator)  # Iterate through the generator

        if not ids:
            return {"message": "No chunk IDs found in Pinecone for this namespace", "namespace": namespace}

        # Print all chunk IDs
        print("Chunk IDs in Pinecone:")
        for chunk_id in ids:
            print(chunk_id)

        # Return chunk IDs in the API response
        return {"namespace": namespace, "chunk_ids": ids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunk IDs from Pinecone: {str(e)}")

