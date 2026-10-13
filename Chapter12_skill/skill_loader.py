"""第十二章 01 的最小技能加载器，整理成模块供综合实战复用。

第 1 层：discover_skills 检查元数据，build_skills_prompt 写进系统提示；
第 2、3 层：create_skill_tools 返回 load_skill 与 read_skill_file 两个工具。
"""

import re
from pathlib import Path

import yaml
from langchain_core.tools import tool


NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def split_skill_file(skill_file: Path) -> tuple[dict, str]:
    """把 SKILL.md 拆成 YAML 元数据字典和 Markdown 正文。"""
    text = skill_file.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{skill_file} 缺少开头的 YAML 元数据")
    _, frontmatter, body = text.split("---\n", 2)
    return yaml.safe_load(frontmatter), body.strip()


def discover_skills(skills_dir: Path) -> dict[str, dict]:
    """扫描技能目录，检查元数据，只保留名称、描述和目录。"""
    skills = {}
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        metadata, _ = split_skill_file(skill_file)
        name = metadata.get("name", "")
        description = metadata.get("description", "")
        problems = []
        if not (len(name) <= 64 and NAME_PATTERN.fullmatch(name)):
            problems.append("name 只能用小写字母、数字和单个连字符，长度 1–64")
        if name != skill_file.parent.name:
            problems.append("name 必须与目录名相同")
        if not 1 <= len(description) <= 1024:
            problems.append("description 长度应为 1–1024")
        if problems:
            raise ValueError(f"{skill_file.parent.name}/SKILL.md：" + "；".join(problems))
        skills[name] = {"description": description, "dir": skill_file.parent}
    return skills


def build_skills_prompt(skills: dict[str, dict]) -> str:
    lines = [
        "## 可用技能",
        "处理用户请求前，先判断它是否与下列技能相关：",
        "- 相关时，先调用 load_skill 加载该技能的完整说明，再按说明执行；",
        "- 无关时直接回答，不要加载技能。",
        "",
    ]
    lines += [f"- {name}：{info['description']}" for name, info in skills.items()]
    return "\n".join(lines)


def create_skill_tools(skills: dict[str, dict]) -> list:
    """为给定的技能集合创建 load_skill 与 read_skill_file 两个工具。"""

    @tool
    def load_skill(skill_name: str) -> str:
        """加载一个技能的完整说明（SKILL.md 正文）。skill_name 使用“可用技能”中的名称。"""
        info = skills.get(skill_name)
        if info is None:
            return f"没有名为 {skill_name} 的技能。可用技能：{', '.join(skills)}"
        _, body = split_skill_file(info["dir"] / "SKILL.md")
        return f"已加载技能 {skill_name}。正文中的相对路径可以用 read_skill_file 读取。\n\n{body}"

    @tool
    def read_skill_file(skill_name: str, path: str) -> str:
        """读取技能目录中的辅助文件。path 使用技能说明中写的相对路径，例如 references/plan-guide.md。"""
        info = skills.get(skill_name)
        if info is None:
            return f"没有名为 {skill_name} 的技能。"
        skill_dir = info["dir"].resolve()
        target = (skill_dir / path).resolve()
        if not target.is_relative_to(skill_dir) or not target.is_file():
            return f"技能 {skill_name} 的目录中没有文件 {path}。"
        return target.read_text(encoding="utf-8")

    return [load_skill, read_skill_file]
