import re
import typing
from discord import Intents
from discord.ext import commands
from github import Github

with open(".env", "r") as f:
    x = f.readlines()

DISCORD_TOKEN = x[0].split("DISCORD_BOT_TOKEN=")[1].strip()
GH_TOKEN = x[1].split("GITHUB_TOKEN=")[1].strip()

intents = Intents.default()
intents.message_content = True

# Discord bot setup
bot = commands.Bot(command_prefix="!", intents=intents)

# GitHub setup
repo = Github(GH_TOKEN).get_repo("dementive/Chronicles-of-Omniluxia")


def create_github_issue(title, content, label):
    # Create the issue on GitHub
    if content:
        issue = repo.create_issue(title=title, body=content)
    else:
        issue = repo.create_issue(title=title)
    if label:
        issue.set_labels(label)


@bot.command()
async def issue(
    ctx,
    title: str,
    label: typing.Optional[str] = None,
    url: typing.Optional[str] = None,
):
    """
    Discord bot command to create a github issue, it is called with the following syntax in a discord channel:
    '!issue <title> <label <discord message url/issue description>'
    """
    if str(ctx.author) != "dementive999":
        await ctx.send("Steve says you are not authorized to use this command.")
        return

    url_pattern = r"https:\/\/discord\.com\/channels\/\d+\/(\d+)\/(\d+)"
    url = "" if url is None else url
    match = re.match(url_pattern, url)

    if match:
        channel_id, message_id = map(int, match.groups())

        # Fetch the channel where the message was posted
        channel = bot.get_channel(channel_id)
        if not channel:
            await ctx.send("Steve couldn't find the channel for that message.")
            return

        try:
            # Fetch the message using its ID
            message = await channel.fetch_message(message_id)
        except Exception as e:
            await ctx.send(f"Steve had an error processing the message: {str(e)}")

        create_github_issue(message.content, url, label)
    else:
        create_github_issue(title, url, label)
        await ctx.send(f"GitHub issue created: {title}")


# Run the bot
bot.run(DISCORD_TOKEN)
