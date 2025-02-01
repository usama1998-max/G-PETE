import streamlit as st
import anthropic
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery, HybridFusion
from weaviate.classes.aggregate import GroupByAggregate
from llama_index.core.node_parser import SentenceSplitter
import requests
import os
import re
import uuid
load_dotenv()

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

class JINA:
    def __init__(self):
        self.__header = {
            "Content-Type": "application/json",
            "Authorization": f"""Bearer {os.environ.get('JINA_AI')}"""
        }
        self.__model = "jina-reranker-v2-base-multilingual"

    def rerank(self, query, docs, top_n: int = 10):
        url = "https://api.jina.ai/v1/rerank"

        data_obj = {
            "model": self.__model,
            "query": query,
            "documents": docs,
            "top_n": top_n
        }

        response = requests.post(url, headers=self.__header, json=data_obj)

        return response.json()


class WeaviateClient(JINA):
    def __init__(self):
        super().__init__()
        self.threshold = 0.50
        self.alpha = 0.5
        self.num_results = 40
        self.ss = SentenceSplitter(chunk_size=350, chunk_overlap=20)

        self.headers = {
            "X-JinaAI-Api-Key": os.environ.get('JINA_AI'),
        }
        self.__auth_config = Auth.api_key(os.environ.get('WEAVIATE_API_KEY'))
        self.weaviate_client = weaviate.connect_to_wcs(
            skip_init_checks=True,
            cluster_url=os.environ.get('WEAVIATE_CLUSTER_URL'),
            auth_credentials=self.__auth_config,
            headers=self.headers
        )

    def get_files(self, class_: str):
        try:
            jeopardy = self.weaviate_client.collections.get(class_)
            response = jeopardy.aggregate.over_all(
                group_by=GroupByAggregate(prop="file_name"),
            )

            graphql_query = f"""
            {{
              Get {{
                {class_}(
                  where: {{
                    operator: Or,
                    operands: [
                      {", ".join([f'{{path: ["file_name"], operator: Equal, valueText: "{name}"}}' for name in [group.grouped_by.value for group in response.groups]])}
                    ]
                  }}
                ) {{
                  file_name
                  uuid
                }}
              }}
            }}
            """
            res = self.weaviate_client.graphql_raw_query(graphql_query)

            # Extract results
            # file_data = [(entry["file_name"], entry["uuid"]) for entry in res.get]
            file_data = []
            unique = []

            for obj in res.get[class_]:
                if obj['uuid'] not in unique:
                    unique.append(obj['uuid'])
                    file_data.append({"uuid": obj['uuid'], "file_name": obj['file_name']})

            return file_data
        except Exception as e:
            return []

    def format_context(self, batch: list):
        return "\n".join(batch)

    def collection_exists(self, class_: str) -> bool:
        return True if self.weaviate_client.collections.exists(class_) is True else False

    def chunk_document(self, corpus: str):
        return self.ss.split_text(corpus)

    def add_collection(self, class_: str) -> None:
        class_obj = {
            "class": class_,
            "description": f"collection for {class_}",
            "vectorizer": "text2vec-jinaai",
            "properties": [
                {
                    "name": "uuid",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-jinaai": {
                            "skip": True,
                        }
                    }
                },
                {
                    "name": "file_name",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-jinaai": {
                            "skip": True,
                        }
                    }
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-jinaai": {
                            "model": "jina-embeddings-v3",
                        }
                    }
                },
            ],
        }

        self.weaviate_client.collections.create_from_dict(class_obj)

    def knowledge_retrieval(self, class_: str, query: str):
        try:
            weaviate_result = []

            filter_query = re.sub(r"\\", "", query).lower()

            collection = self.weaviate_client.collections.get(class_)
            response = collection.query.hybrid(
                query=filter_query,
                # filters=(
                #         Filter.by_property("entity").equal(entity) &
                #         Filter.by_property("user_id").equal(user_id)
                # ),

                # query_properties=["collection_name"],
                alpha=self.alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=MetadataQuery(score=True),
                return_properties=["content", "file_name", "uuid"],
                limit=self.num_results
            )

            if not response or response != [] or hasattr(response, 'objects') is True:
                result = response.objects

                if result is not None:
                    for chunk in result:

                        relevance_score = round(float(chunk.metadata.score), 3)

                        if relevance_score >= self.threshold:
                            weaviate_result.append(chunk.properties['content'])

                return weaviate_result
            else:
                # logger.info("NO RESULTS FOUND...")
                return []
        except Exception as e:
            # logger.error(f"CLASS LLMHYBRID -> USER VEC: {e}")
            print(e)
            st.error("Something went wrong with Knowledge Base!")
            return []

    def knowledge_reranker(self, query: str, batch: list, top_k: int=10):
        try:
            if not batch or batch == []:
                return batch

            ranked_results = []

            results = self.rerank(query=query, docs=batch, top_n=top_k)

            for document in results['results']:
                if float(document['relevance_score']) >= 0.400:
                    ranked_results.append(document['document']['text'])

            return ranked_results

        except Exception as e:
            print(e)
            st.error("Something went wrong with Knowledge Base!")
            return []

    def add_batch_uuid(self,
                       class_: str,
                       batch: list,
                       uuid_: str = "",
                       file_name: str = "") -> str:
        try:

            data_objs = [
                {
                    "content": str(chunk),
                    "uuid": str(uuid_),
                    "file_name": str(file_name)
                } for chunk in batch
            ]

            collection = self.weaviate_client.collections.get(class_)

            with collection.batch.fixed_size(batch_size=100) as batch:
                for data_row in data_objs:
                    batch.add_object(
                        properties=data_row,
                    )

            return str(uuid_)
        except Exception as e:
            # logger.error("ADD BATCH UUID: ")
            # logger.error(e)
            print(e)
            st.error("Something went wrong with Knowledge Base!")
            return ""

    def remove_uuid(self, class_: str, uuid_: str) -> bool:
        try:
            collection = self.weaviate_client.collections.get(class_)
            collection.data.delete_many(
                where=Filter.by_property("uuid").equal(uuid_)
            )

            return True

        except Exception as e:
            print(e)
            st.error("Something went wrong with Knowledge Base!")
            return False


def system_prompt(query: str, context: str = "") -> str:
    return f"""Your name is G-PETE, You work as an attorney. 
    You are not an AI but a human. 
    Your response should be friendly and helpful. 
    Help the following user to answer the question.
    Do not add regards.
    
    You can use the following context to answer the question if relevant:
    {context}
    
    USER: {query}
    🤖 G-PETE: ...
    """


def stream_response(response):
    for event in response:
        if hasattr(event, 'delta') is True:
            if hasattr(event.delta, 'text') is True:
                yield event.delta.text


knowledge_base = WeaviateClient()

STATUS_LIVE = knowledge_base.weaviate_client.is_live()
KNOWLEDGE_BASE_NAME = "AttorneyInfo"

if STATUS_LIVE is True:
    if knowledge_base.collection_exists(KNOWLEDGE_BASE_NAME) is False:
        knowledge_base.add_collection(KNOWLEDGE_BASE_NAME)
        st.success("🤖 G-PETE: There was no schema, dont worry i created it")

client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API'))


# SIDE BAR
st.sidebar.markdown(f"""<p>KNOWLEDGE BASE STATUS: {'<span style="color:green;"> OK!</span>' if STATUS_LIVE is True else '<span style="color:red;"> NOT OK!</span>'}</p>""", unsafe_allow_html=True)
st.sidebar.title("🧠 Knowledge Base")
st.sidebar.text("🤖 G-PETE: You can only upload TXT files for now...")


uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt"], key="upload_file", accept_multiple_files=False)


# st.success("✅ Operation successful!")
# st.warning("⚠️ Warning! Check your input.")
# st.error("❌ An error occurred.")
# st.info("ℹ️ This is an informational message.")


if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")

        docs = knowledge_base.chunk_document(content)

        unique_id = str(uuid.uuid4())
        knowledge_base.add_batch_uuid(KNOWLEDGE_BASE_NAME, docs, unique_id, uploaded_file.name)

        st.sidebar.success(f"🤖 G-PETE: I learned everything from {uploaded_file.name} ✅")
        st.session_state.uploaded_file = None


files = knowledge_base.get_files(KNOWLEDGE_BASE_NAME)


for i, file_obj in enumerate(files):
    if "file_name" in file_obj:
        col1, col2 = st.sidebar.columns([3, 1])

        with col1:
            st.sidebar.markdown(
                f"""<div style="border: 2px solid white; padding: 10px; margin-top:5px; border-radius:10px;"> {file_obj['file_name']} </div>""",
                unsafe_allow_html=True
            )

        with col2:
            if st.sidebar.button("Remove", key=f"process_{i}"):  # Unique key
               knowledge_base.remove_uuid(KNOWLEDGE_BASE_NAME, file_obj['uuid'])
               files = knowledge_base.get_files(KNOWLEDGE_BASE_NAME)

st.title("🤖 G-PETE")
st.text("Hi there! I'm G-PETE, and I'm an attorney who's been practicing law for several years. I really enjoy helping people navigate their legal questions and concerns. I aim to provide clear, practical legal guidance while keeping things friendly and approachable - law can be complicated enough without making it more intimidating! Lets have a conversation shall we")
text = st.text_input("Start a conversation...").lower()

# response.content[0].text
output = st.empty()


if text:
    vectors = knowledge_base.knowledge_retrieval(KNOWLEDGE_BASE_NAME, text)
    knowledge = knowledge_base.knowledge_reranker(text, vectors, 10)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{'role': 'user', 'content': system_prompt(query=text, context=knowledge_base.format_context(knowledge))}],
        stream=True
    )
    st.markdown("### Found related Knowledge in the Document:")
    st.write(vectors)

    st.markdown("### I selected the following context related to question:")
    st.write(knowledge)
    output.write_stream(stream_response(response))
    knowledge_base.weaviate_client.close()

