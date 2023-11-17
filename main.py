from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import llms
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os 
os.environ['OPENAI_API_KEY'] = 'dummy_key'
from paperqa import Docs
from datetime import datetime
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()
from paperqa import Docs, Answer, PromptCollection
from langchain.prompts import PromptTemplate
import pickle


my_qaprompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question '{question}' "
    "Use the context below if helpful. "
    "You can cite the context using the key "
    "like (Example2012). "
    "If there is insufficient context, write a poem "
    "about how you cannot answer.\n\n"
    "Context: {context}\n\n"
)
prompts=PromptCollection(qa=my_qaprompt)

print(datetime.now())
print("Loading model")

model_id = "bigscience/bloom-7b1"  # needs more than 12GB of RAM. Needs 17GB of GPU VRAM
model_id = "bigscience/bloom-560m"  # needs more than 12GB of RAM
model = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 100},
        )

embeddings = HuggingFaceEmbeddings(model_name=model_id)

print(datetime.now())
print("Loading documents")

# TODO: use a vectore database to store vectorized documents
# TODO: if a document is added, add it after loading the pickel object
# load docs from pickle object
if (os.path.exists("my_docs.pkl") and False):
    with open("my_docs.pkl", "rb") as f:
        docs = pickle.load(f)
else:
    docs = Docs()
    docs.llm = model
    docs.summary_llm = model
    docs.embeddings = embeddings
    docs.prompts = prompts
    docs.add("docs/s0103-21862012000200002.pdf", chunk_chars=400, docname="iop")

    # # save into pickle object
    # with open("my_docs.pkl", "wb") as f:
    #     pickle.dump(docs, f)

print(datetime.now())
print("Answering question")

answer = docs.query("Qual foi o papel da alphabetizaçao ?")
print(answer.formatted_answer)
