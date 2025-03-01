import sys
import os

from ayaka_utils.Classes.Timer import Timer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EmoTensorGen_Service import ProcessEmotionalTransient

with Timer():
    ProcessEmotionalTransient(
        PerspectiveUser="uid=1", # This should stay the same for all messages, and typically will be the perspective of the AI.
        ThisMessage="ねえ、アシスタント、今日の天気はどう？", # This is the current message to process
        SpeakerUser="uid=5", # This is the user who is speaking
        Emoters=[ # Always list users in tensoral (numerical) order.
            {
                "emoter_user": "uid=1",
                "external_context": "アシスタント is a personal AI Agent that is currently assisting 花子.",
                "targets": [
                    {
                        "this_target": "uid=1-uid=5"  # This user -> First other user
                    }
                ]
            },
            {
                "emoter_user": "uid=5",
                "external_context": "花子 is checking the weather for today.",
                "targets": [
                    {
                        "this_target": "uid=5-uid=1"  # This user -> First other user
                    }
                ]
            },
        ],
        EmbedContext=( # ToDo: get text embedding query results from the Embed-Host
            "アシスタント enjoys helping 花子. They have previously had pleasant and enjoyable interactions. "
            "Previously, 花子 forgot to bring her umbrella to work and got wet in the rain on the walk to the station after work. "
            "花子 works in Ginza and commutes from Saitama. "
            "\n"
        ),
        #EmoTensorContextualizedFilePath="./tests/EmoTensorDB/EmoTensor-TEST.etsc",
        EmoTensorContextualizedFilePath="/EmoTensor/data/Sliced-Contextualized/EmoTensor-TEST.etsc",
        # ToDo: get current users from the user DB and pass them in here.
        # ToDo: also, add a handler for if both users have the same name.
        ConversationUsers=[
            {
                "id": "uid=1",
                "name": "アシスタント",
                "phonetic_jp": "アシスタント",
                "phonetic_en": "Assistant",
                "preferred_name": "アシスタント",
                "gender": "agender",
                "pronouns": "{ 'Jp':{ 'fpp': '私' }, 'En': { 'Sub': 'it', 'Obj': 'it', 'Pos': 'its', 'Ref': 'itself' } }",
                "bio": "Generic Personal AI Assistant"
            },
            {
                "id": "uid=5",
                "name": "山田・花子",
                "phonetic_jp": "やまだ・はなこ",
                "phonetic_en": "Yamada Hanako",
                "preferred_name": "花子",
                "gender": "woman",
                "pronouns": "{ 'Jp':{ 'fpp': '私' }, 'En': { 'Sub': 'she', 'Obj': 'her', 'Pos': 'hers', 'Ref': 'herself' } }",
                "bio": "Placeholder User. 花子 represents any user of the service."
            }
        ],
        TestMode=False
    )
