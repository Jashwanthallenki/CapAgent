import ast

def extract_tool_prompt(file_path):
    docstrings_list = []

    # Open and parse the file
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                class_docstring = ast.get_docstring(node)
                init_signature = None
                init_docstring = None

                # Look for __init__ method
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                        params = []
                        args_with_defaults = zip(
                            child.args.args[-len(child.args.defaults):], child.args.defaults
                        )
                        args_no_defaults = child.args.args[:len(child.args.args) - len(child.args.defaults)]

                        # Process arguments without defaults
                        for arg in args_no_defaults:
                            param = arg.arg
                            if arg.annotation:
                                param += f": {ast.unparse(arg.annotation)}"
                            params.append(param)

                        # Process arguments with defaults
                        for arg, default in args_with_defaults:
                            param = arg.arg
                            if arg.annotation:
                                param += f": {ast.unparse(arg.annotation)}"
                            param += f" = {ast.unparse(default)}"
                            params.append(param)

                        # Add *args and **kwargs if present
                        if child.args.vararg:
                            params.append(f"*{child.args.vararg.arg}")
                        if child.args.kwarg:
                            params.append(f"**{child.args.kwarg.arg}")

                        init_signature = f"def __init__({', '.join(params)})"
                        if child.returns:
                            init_signature += f" -> {ast.unparse(child.returns)}"
                        init_docstring = ast.get_docstring(child)

                # Combine class and __init__ details
                class_entry = f"class {class_name}:\n"
                if class_docstring:
                    class_entry += f"\"\"\"\n{class_docstring}\n\"\"\"\n"
                if init_signature and init_docstring:
                    class_entry += f"    {init_signature}:\n    \"\"\"\n{init_docstring}\n\"\"\"\n"
                docstrings_list.append(class_entry)

            elif isinstance(node, ast.FunctionDef):
                # Handle standalone functions
                name = node.name
                params = []
                args_with_defaults = zip(
                    node.args.args[-len(node.args.defaults):], node.args.defaults
                )
                args_no_defaults = node.args.args[:len(node.args.args) - len(node.args.defaults)]

                for arg in args_no_defaults:
                    param = arg.arg
                    if arg.annotation:
                        param += f": {ast.unparse(arg.annotation)}"
                    params.append(param)

                for arg, default in args_with_defaults:
                    param = arg.arg
                    if arg.annotation:
                        param += f": {ast.unparse(arg.annotation)}"
                    param += f" = {ast.unparse(default)}"
                    params.append(param)

                if node.args.vararg:
                    params.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    params.append(f"**{node.args.kwarg.arg}")

                signature = f"def {name}({', '.join(params)})"
                if node.returns:
                    signature += f" -> {ast.unparse(node.returns)}"
                docstring = ast.get_docstring(node)
                if docstring:
                    function_entry = f"{signature}:\n\"\"\"\n{docstring}\n\"\"\"\n"
                    docstrings_list.append(function_entry)

    return "\n".join(docstrings_list)

# Example usage
if __name__ == "__main__":
    result = extract_tool_prompt("capagent/tools.py")
    print(result)
