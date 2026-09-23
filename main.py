from importlib.metadata import PackageNotFoundError, version

def main():
    try:
        # 传入你通过 uv 安装的包名（例如 langchain）
        pkg_version = version("langchain")
        print(f"langchain 版本: {pkg_version}")
    except PackageNotFoundError:
        print("当前环境中未安装该包。")


if __name__ == "__main__":
    main()
