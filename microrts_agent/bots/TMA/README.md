# MicroRTS

A submission for the MicroRTS competition hosted at the CoG2024.

The TMA bot (Tactical Manager AI) consists of a main script that analyzes the current situation of the match (map size, resources, distance to the enemy, unit distribution) and uses those informations to switch between four main strategies in real time.

The four strategies (two defensive and two offensive) are meant to have an hard coded general flow, but they are also flexible, with decision points and thresholds being modified and adapted in real time by TMA itself. This allows for decisions to be taken rapidly, but to also have them adapt to the current situation.

The repository includes TMA (the main code of the Bot) and a folder with the supporting scripts. There are two versions of this folder, with the first one being an outdated and useless version of certain scripts. If the bot does not work on your local machine, try to remove the outdated "strategies" folder.

For any question, you can contact me on my personal mail: ale.mazza1999@gmail.com
