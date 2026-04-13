import os
import socket
from pathlib import Path

import docker
import fire
import typer
import openai
from dotenv import load_dotenv

# 在导入其他模块前加载 .env（使用绝对路径）
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

from cloudgpt_aoai import get_chat_completion, get_openai_client, async_get_openai_client
from typing_extensions import Annotated

from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import cleanup_container


def check_docker_status() -> None:
    container = None
    try:
        client = docker.from_env()
        client.images.pull("hello-world")
        container = client.containers.run("hello-world", detach=True)
        logs = container.logs().decode("utf-8")
        print(logs)
        logger.info(f"The docker status is normal")
    except docker.errors.DockerException as e:
        logger.error(f"An error occurred: {e}")
        logger.warning(
            f"Docker status is exception, please check the docker configuration or reinstall it. Refs: https://docs.docker.com/engine/install/ubuntu/."
        )
    finally:
        cleanup_container(container, "health check")


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_and_list_free_ports(start_port=19899, max_ports=10) -> None:
    is_occupied = is_port_in_use(port=start_port)
    if is_occupied:
        free_ports = []
        for port in range(start_port, start_port + max_ports):
            if not is_port_in_use(port):
                free_ports.append(port)
        logger.warning(
            f"Port 19899 is occupied, please replace it with an available port when running the `rdagent ui` command. Available ports: {free_ports}"
        )
    else:
        logger.info(f"Port 19899 is not occupied, you can run the `rdagent ui` command")


def test_chat(chat_model, chat_api_key, chat_api_base):
    logger.info(f"🧪 Testing chat model: {chat_model}")
    try:
        # Use cloudgpt_aoai helper for unified API handling
        if chat_api_key:
            # For custom API keys, temporarily set them
            openai.api_key = chat_api_key
            if chat_api_base:
                openai.api_base = chat_api_base
            # Create client with custom credentials
            client = openai.OpenAI(api_key=chat_api_key, base_url=chat_api_base)
            response = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello!"}],
            )
        else:
            # Use Azure AD authentication via cloudgpt_aoai
            response = get_chat_completion(
                messages=[{"role": "user", "content": "Hello!"}],
                model=chat_model,
            )
        logger.info(f"✅ Chat test passed.")
        return True
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        return False


def test_embedding(embedding_model, embedding_api_key, embedding_api_base):
    logger.info(f"🧪 Testing embedding model: {embedding_model}")
    try:
        if embedding_api_key:
            # For custom API keys, create OpenAI client directly
            client = openai.OpenAI(api_key=embedding_api_key, base_url=embedding_api_base)
        else:
            # Use Azure AD authentication via cloudgpt_aoai
            client = get_openai_client()
        
        # Use modern client API
        res = client.embeddings.create(
            model=embedding_model,
            input="Hello world!",
        )
        logger.info("✅ Embedding test passed.")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return False


def env_check():
    if "BACKEND" not in os.environ:
        logger.warning(
            f"We did not find BACKEND in your configuration, please add it to your .env file. "
            f"You can run a command like this: `dotenv set BACKEND rdagent.oai.backend.LiteLLMAPIBackend`"
        )

    # Determine authentication method: API Keys or Azure AD
    use_azure_ad = False
    chat_api_key = None
    chat_api_base = None
    embedding_api_key = None
    embedding_api_base = None
    chat_model = os.getenv("CHAT_MODEL")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    if "DEEPSEEK_API_KEY" in os.environ:
        chat_api_key = os.getenv("DEEPSEEK_API_KEY")
        embedding_api_key = os.getenv("LITELLM_PROXY_API_KEY")
        embedding_api_base = os.getenv("LITELLM_PROXY_API_BASE")
        if "DEEPSEEK_API_BASE" in os.environ:
            chat_api_base = os.getenv("DEEPSEEK_API_BASE")
        elif "OPENAI_API_BASE" in os.environ:
            chat_api_base = os.getenv("OPENAI_API_BASE")
    elif "OPENAI_API_KEY" in os.environ:
        chat_api_key = os.getenv("OPENAI_API_KEY")
        chat_api_base = os.getenv("OPENAI_API_BASE")
        embedding_api_key = chat_api_key
        embedding_api_base = chat_api_base
    else:
        # Default to Azure AD authentication via cloudgpt_aoai
        logger.info("No API keys found, using Azure AD authentication")
        use_azure_ad = True

    logger.info("🚀 Starting test...\n")
    
    # Test embedding model
    result_embedding = test_embedding(
        embedding_model=embedding_model, 
        embedding_api_key=embedding_api_key, 
        embedding_api_base=embedding_api_base
    )
    
    # Test chat model
    result_chat = test_chat(
        chat_model=chat_model, 
        chat_api_key=chat_api_key, 
        chat_api_base=chat_api_base
    )

    if result_chat and result_embedding:
        logger.info("✅ All tests completed.")
    else:
        logger.error("❌ One or more tests failed. Please check credentials or model support.")


def health_check(
    check_env: Annotated[bool, typer.Option("--check-env/--no-check-env", "-e/-E")] = True,
    check_docker: Annotated[bool, typer.Option("--check-docker/--no-check-docker", "-d/-D")] = True,
    check_ports: Annotated[bool, typer.Option("--check-ports/--no-check-ports", "-p/-P")] = True,
):
    """
    Run the RD-Agent health check:
    - Check if Docker is available
    - Check that the default ports are not occupied
    - (Optional) Check that the API Key and model are configured correctly.

    Args:
        check_env (bool): Whether to check API Key and model configuration.
        check_docker (bool): Checks if Docker is installed and running.
        check_ports (bool): Whether to check if the default port (19899) is occupied.
    """
    check_any = False

    if check_env:
        check_any = True
        env_check()
    if check_docker:
        check_any = True
        check_docker_status()
    if check_ports:
        check_any = True
        check_and_list_free_ports()

    if not check_any:
        logger.warning("⚠️ All health check items are disabled. Please enable at least one check.")


if __name__ == "__main__":
    typer.run(health_check)
