import inquirer
from yourdfpy import Link


def select_link(links: list[Link]) -> Link:
    questions = [
        inquirer.List(
            "link",
            message="Select a link",
            choices=[link.name for link in links],
        )
    ]
    answers = inquirer.prompt(questions)
    for link in links:
        if link.name == answers["link"]:
            return link
    raise ValueError(f"Link {answers['link']} not found")
