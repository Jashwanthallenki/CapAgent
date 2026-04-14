import os, sys, ast, re, subprocess, tempfile
import requests
from io import BytesIO
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "autogen"))

from autogen.coding import CodeBlock
from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor
from capagent.config import IMAGE_SERVER_DOMAIN_NAME

'''parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
'''

parent_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(parent_dir, ".."))  # ...\CapAgent
project_root = parent_dir

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ---------------------------
# ✅ Custom Local Executor
# ---------------------------
class LocalCommandLineCodeExecutor:
    def __init__(self, work_dir=None):
        #self.work_dir = work_dir or tempfile.mkdtemp()
        self.work_dir = work_dir or project_root


    def execute(self, code, language="python"):
        if language != "python":
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Unsupported language: {language}",
                "output_files": [],
                "output": ""
            }
        filename = os.path.join(repo_root, "temp_code.py")
        #filename = os.path.join(self.work_dir, "temp_code.py")
        with open(filename, "w") as f:
            f.write(code)

        try:
            result = subprocess.run(
                ["python", filename],
                capture_output=True,
                text=True,
                cwd=repo_root,
                timeout=60
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": [],
                "output": result.stdout + ("\n" + result.stderr if result.stderr else "")
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "Execution timed out.",
                "output_files": [],
                "output": "Execution timed out."
            }


# ---------------------------
# ✅ CodeExecutor wrapper
# ---------------------------
class CodeExecutor:
    def __init__(self, working_dir: str = "", use_tools: bool = False, use_docker: bool = False):
        self.working_dir = working_dir or "."
        os.makedirs(self.working_dir, exist_ok=True)
        self.use_docker = use_docker

        if use_docker:
            # 🚀 Docker-based Jupyter executor
            self.server = DockerJupyterServer()
            print(f"Docker Jupyter server created: {self.server}")
            self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir)
            print("Jupyter executor ready")
        else:
            # 🖥️ Local executor
            self.server = None
            self.executor = LocalCommandLineCodeExecutor(work_dir=self.working_dir)
            print("Local executor ready")

        # Initialize environment
        self.init_env(use_tools)
        print("Environment initialized")

    '''def loading_images(self, image_paths):
        code = ""
        os.makedirs(".tmp", exist_ok=True)

        print("[DEBUG] Starting image loading...")
        print(f"[DEBUG] Received {len(image_paths)} image paths.")

        for idx, path in enumerate(image_paths):
            local_file = os.path.join(project_root, f"image_{idx+1}.png")
            print(f"\n[DEBUG] Processing image {idx+1}: {path}")

            if path.startswith("http://") or path.startswith("https://"):
                print("[DEBUG] Detected URL, downloading image...")
                try:
                    response = requests.get(path)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    print("[DEBUG] Successfully downloaded and converted image.")
                except Exception as e:
                    print(f"[ERROR] Failed to download image {path}: {e}")
                    continue
            else:
                print("[DEBUG] Detected local path, opening image...")
                try:
                    image = Image.open(path).convert("RGB")
                    print("[DEBUG] Successfully opened local image.")
                except Exception as e:
                    print(f"[ERROR] Failed to open local image {path}: {e}")
                    continue

            # Save locally so we always have a consistent file reference
            image.save(local_file)
            print(f"[DEBUG] Saved image as {local_file}")

            # Generate code
            #code += f"""image_{idx+1} = Image.open("{local_file}").convert("RGB")\n"""
            #code += f"""image_data_{idx+1} = ImageData(image_{idx+1}, image_url=f"{IMAGE_SERVER_DOMAIN_NAME}/{local_file}", local_path="{local_file}")\n"""
            #code += f"""image_{idx+1} = Image.open("{local_file}").convert("RGB")\n"""
            #code += f"""print("Loaded image_{idx+1} with size", image_{idx+1}.size)\n"""
            code += f"""image_{idx+1} = Image.open(r"{local_file}").convert("RGB")\n"""
            code += f"""print("Loaded image_{idx+1} with size", image_{idx+1}.size)\n"""
            print(f"[DEBUG] Added code for image_{idx+1}")

        print("\n[DEBUG] Finished processing all images.")
        return self.execute(code)

'''


    def loading_images(self, image_paths):
        code = ""
        os.makedirs(".tmp", exist_ok=True)

        # ✅ Ensure we have a safe outputs/images directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(project_root, "outputs", "images")
        os.makedirs(output_dir, exist_ok=True)

        print("[DEBUG] Starting image loading...")
        print(f"[DEBUG] Received {len(image_paths)} image paths.")

        for idx, path in enumerate(image_paths):
            local_file = os.path.join(output_dir, f"image_{idx+1}.png")
            print(f"\n[DEBUG] Processing image {idx+1}: {path}")

            if path.startswith("http://") or path.startswith("https://"):
                print("[DEBUG] Detected URL, downloading image...")
                try:
                    response = requests.get(path)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    print("[DEBUG] Successfully downloaded and converted image.")
                except Exception as e:
                    print(f"[ERROR] Failed to download image {path}: {e}")
                    continue
            else:
                print("[DEBUG] Detected local path, opening image...")
                try:
                    image = Image.open(path).convert("RGB")
                    print("[DEBUG] Successfully opened local image.")
                except Exception as e:
                    print(f"[ERROR] Failed to open local image {path}: {e}")
                    continue

            # ✅ Save locally inside outputs/images
            image.save(local_file)
            print(f"[DEBUG] Saved image as {local_file}")

            # ✅ Generate code that uses this absolute path
            code += f"""image_{idx+1} = Image.open(r"{local_file}").convert("RGB")\n"""
            code += f"""print("Loaded image_{idx+1} with size", image_{idx+1}.size)\n"""
            print(f"[DEBUG] Added code for image_{idx+1}")

        print("\n[DEBUG] Finished processing all images.")
        return self.execute(code)


    def result_processor(self, result):
        def parse_error_message(error):
            list_start_index = error.find("['")
            initial_error = error[:list_start_index].strip()
            traceback_list_str = error[list_start_index:]
            try:
                traceback_list = ast.literal_eval(traceback_list_str)
            except SyntaxError as e:
                traceback_list = []
            ansi_escape = re.compile(r'\x1b\[.*?m')
            traceback_list = [ansi_escape.sub('', line) for line in traceback_list]
            return initial_error + "\n\n" + "\n".join(traceback_list)

        # Local executor returns dict, Jupyter returns object
        exit_code = getattr(result, "exit_code", result.get("exit_code", 1))
        file_paths = getattr(result, "output_files", result.get("output_files", []))
        output = getattr(result, "output", result.get("output", ""))
        output_lines = output.split("\n")

        if len(file_paths) > 0:
            output_lines = output_lines[:-2 * len(file_paths)]

        if exit_code == 0:
            new_str = ""
            image_idx = 0
            for line in output_lines:
                if line.startswith("<PIL.") and image_idx < len(file_paths):
                    new_str += f"<img src='{file_paths[image_idx]}'>"
                    image_idx += 1
                else:
                    new_str += line
                new_str += "\n"
            return exit_code, new_str, file_paths
        else:
            return exit_code, parse_error_message(output), file_paths

    def execute(self, code: str):
        print("Code::",code)
        prelude = "from PIL import Image\n"
        code = prelude + code
        if self.use_docker:
            # For Docker Jupyter executor
            self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
            execution_result = self.executor.execute_code_blocks([CodeBlock(language="python", code=code)])
        else:
            # Local executor
            execution_result = self.executor.execute(code)
            print("execution result",execution_result)

        return self.result_processor(execution_result)

    def init_env(self, use_tools):
        init_code = (
            "import sys\n"
            "from PIL import Image\n"
            "from IPython.display import display\n"
            f"parent_dir = '{project_root}'\n"
            "if project_root not in sys.path:\n"
            "    sys.path.insert(0, project_root)\n"
            #"from capagent.utils import ImageData\n"
        )
        print("Parent_dir",parent_dir)
        if use_tools:
            init_code += "from capagent.tools import *\n"

        init_resp = self.execute(init_code)
        print(init_resp[1])

    def cleanup(self):
        if self.use_docker and hasattr(self.server, "stop"):
            self.server.stop()
            print("Docker Jupyter server stopped")
        else:
            print("Cleanup skipped (local executor)")



'''class LocalCommandLineCodeExecutor:
    def __init__(self, work_dir=None):
        self.work_dir = work_dir or tempfile.mkdtemp()

    def execute(self, code, language="python"):
        """
        Executes given code in the local environment.
        Supports Python, Bash (shell), etc.
        """
        if language == "python":
            filename = os.path.join(self.work_dir, "temp_code.py")
            with open(filename, "w") as f:
                f.write(code)
            try:
                result = subprocess.run(
                    ["python", filename],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                return {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired:
                return {"exit_code": -1, "stdout": "", "stderr": "Execution timed out."}

        elif language == "bash":
            try:
                result = subprocess.run(
                    code,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.work_dir,
                    timeout=60
                )
                return {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired:
                return {"exit_code": -1, "stdout": "", "stderr": "Execution timed out."}

        else:
            return {"exit_code": -1, "stdout": "", "stderr": f"Unsupported language: {language}"}
'''
'''import os, sys, ast, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "autogen"))
from autogen.coding import CodeBlock
from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor
from capagent.config import IMAGE_SERVER_DOMAIN_NAME

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class CodeExecutor:
    def __init__(self, working_dir: str = "", use_tools: bool = False):
        self.working_dir = working_dir or "."
        os.makedirs(self.working_dir, exist_ok=True)

        # Set up Docker Jupyter Server
        self.server = DockerJupyterServer()
        print("server:", self.server)
        
        #self.server.start()
       
        # Set up Jupyter executor
        self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir)
        print("over 1")
        # Initialize environment
        self.init_env(use_tools)
        print("over 2")

        # Set up Docker Jupyter Server (auto-starts, no .start() or .gateway_url)
        self.server = DockerJupyterServer()
        print(f"Docker Jupyter server created: {self.server}")

        # Set up Jupyter executor
        self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir)
        print("Jupyter executor ready")

        # Initialize environment
        self.init_env(use_tools)
        print("Environment initialized")

    def loading_images(self, image_paths):
        code = ""
        for idx, path in enumerate(image_paths):
            code += f"""image_{idx+1} = Image.open("{path}").convert("RGB")\n"""
            code += f"""image_{idx+1}.save(".tmp/image_{idx+1}.png")\n"""
            code += f"""image_data_{idx+1} = ImageData(image_{idx+1}, image_url=f"{IMAGE_SERVER_DOMAIN_NAME}/.tmp/image_{idx+1}.png", local_path=".tmp/image_{idx+1}.png")\n"""
            print("over 3")
        return self.execute(code)

    def result_processor(self, result):
        def parse_error_message(error):
            list_start_index = error.find("['")
            initial_error = error[:list_start_index].strip()
            traceback_list_str = error[list_start_index:]
            try:
                traceback_list = ast.literal_eval(traceback_list_str)
            except SyntaxError as e:
                print("Error parsing the list: ", e)
                traceback_list = []
            ansi_escape = re.compile(r'\x1b\[.*?m')
            traceback_list = [ansi_escape.sub('', line) for line in traceback_list]
            print("over 4")
            return initial_error + "\n\n" + "\n".join(traceback_list)

        exit_code = result.exit_code
        file_paths = result.output_files
        output_lines = result.output.split("\n")

        if len(file_paths) > 0:
            output_lines = output_lines[:-2 * len(file_paths)]

        if exit_code == 0:
            new_str = ""
            image_idx = 0
            for line in output_lines:
                if line.startswith("<PIL."):
                    new_str += f"<img src='{file_paths[image_idx]}'>"
                    image_idx += 1
                else:
                    new_str += line
                new_str += "\n"
            return exit_code, new_str, file_paths
        else:
            return exit_code, parse_error_message(result.output), file_paths

    def execute(self, code: str):
        self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
        execution_result = self.executor.execute_code_blocks([
            CodeBlock(language="python", code=code)
        ])
        return self.result_processor(execution_result)

    def init_env(self, use_tools):
        init_code = (
            "import sys\n"
            "from PIL import Image\n"
            "from IPython.display import display\n"
            f"parent_dir = '{parent_dir}'\n"
            "if parent_dir not in sys.path:\n"
            "    sys.path.insert(0, parent_dir)\n"
        )
        if use_tools:
            init_code += "from tools import *\n"

        init_resp = self.execute(init_code)
        print(init_resp[1])

    def cleanup(self):
        self.server.stop()
        print("over final")
        if hasattr(self.server, "stop"):
            self.server.stop()
            print("Docker Jupyter server stopped")
        else:
            print("Cleanup skipped (server has no .stop())")'''

