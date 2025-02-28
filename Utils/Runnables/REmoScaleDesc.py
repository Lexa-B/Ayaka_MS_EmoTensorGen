# Python
import sys
import os
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain
from langchain.schema.runnable import RunnableLambda

EmoDescFile = "./Configs/EmoList_Plutchik.xml"

EmoScaleLabelsDefs = ast.literal_eval(open("./Configs/EmoScaleLabels.jsonc", "r").read())


################################################################################
## ToDo:
##  - Finish i18n support
################################################################################


################################################################################
## Define the scales for the intensity, valence, and arousal of emotions within this system to be used in the prompt

def BuildReadableEmoScaleDesc(EmoScaleLabels, PromptGenLang=None) -> str:
    """
    Builds a readable description of the emotion scales for the given language.
    """
    # If PromptGenLang is not set by an argument, use the language set in EmoScaleLabels config file
    if PromptGenLang is None:
        if EmoScaleLabels['PromptGenLang'] is not None:
            PromptGenLang = EmoScaleLabels['PromptGenLang']
        else:
            raise ValueError("PromptGenLang is not set")
    PromptGenLang = PromptGenLang.lower()

    # Build the description of the emotion scales
    EmoScalePromptDesc = (
        "Here are the scales for the intensity, valence, and arousal of emotions within this system:\n"
        "\nHere is how intensity is defined:\n"
        f"* Intensity: {EmoScaleLabels['Intensity']['Definition']}\n"
        f"* Intensity Scale: {EmoScaleLabels['Intensity']['ScaleDesc']}\n"
    )

    # Build the description of the intensity scale
    for i in range(len(EmoScaleLabels['Intensity']['Scale'])):
        if i < len(EmoScaleLabels['Intensity']['Scale']) - 1:
            EmoScalePromptDesc += f"  * Values: {EmoScaleLabels['Intensity']['Scale'][i]} - {EmoScaleLabels['Intensity']['Scale'][i+1]}: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Intensity']['Labels_Ja'][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Intensity']['Labels_En'][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")
        else:
            EmoScalePromptDesc += f"  * Values: {EmoScaleLabels['Intensity']['Scale'][i]} - 1: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Intensity']['Labels_Ja'][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Intensity']['Labels_En'][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")

    EmoScalePromptDesc += (
        "\nHere is how valence is defined:\n"
        f"* Valence: {EmoScaleLabels['Valence']['Definition']}\n"
        f"* Valence Scale: {EmoScaleLabels['Valence']['ScaleDesc']}\n"
    )

    # Build the description of the valence scale for the negative valence
    for i in range(len(EmoScaleLabels['Valence']['Scale'][0])):
        if i == 0:
            EmoScalePromptDesc += f"  * Values: -1.0 - {EmoScaleLabels['Valence']['Scale'][0][i]}: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_Ja'][0][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_En'][0][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")
        else:
            EmoScalePromptDesc += f"  * Values: {EmoScaleLabels['Valence']['Scale'][0][i-1]} - {EmoScaleLabels['Valence']['Scale'][0][i]}: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_Ja'][0][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_En'][0][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")
                
    # Build the description of the valence scale for the positive valence
    for i in range(len(EmoScaleLabels['Valence']['Scale'][1])):
        if i < len(EmoScaleLabels['Valence']['Scale'][1]) - 1:
            EmoScalePromptDesc += f"  * Values: {EmoScaleLabels['Valence']['Scale'][1][i]} - {EmoScaleLabels['Valence']['Scale'][1][i+1]}: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_Ja'][1][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_En'][1][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")
        else:
            EmoScalePromptDesc += f"  * Values: {EmoScaleLabels['Valence']['Scale'][1][i]} - 1.0: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_Ja'][1][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Valence']['Labels_En'][1][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")
                
    EmoScalePromptDesc += (
        "\nHere is how arousal is defined:\n"
        f"* Arousal: {EmoScaleLabels['Arousal']['Definition']}\n"
        f"* Arousal Scale: {EmoScaleLabels['Arousal']['ScaleDesc']}\n"
    )

    # Build the description of the arousal scale
    for i in range(len(EmoScaleLabels['Arousal']['Scale'])):
        if i < len(EmoScaleLabels['Arousal']['Scale']) - 1:
            EmoScalePromptDesc += f"  * Values: {EmoScaleLabels['Arousal']['Scale'][i]} - {EmoScaleLabels['Arousal']['Scale'][i+1]}: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Arousal']['Labels_Ja'][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Arousal']['Labels_En'][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")
        else:
            EmoScalePromptDesc += f"  * Values: {EmoScaleLabels['Arousal']['Scale'][i]} - 1: "
            if PromptGenLang == 'ja':
                EmoScalePromptDesc += f"{EmoScaleLabels['Arousal']['Labels_Ja'][i]}\n"
            elif PromptGenLang == 'en':
                EmoScalePromptDesc += f"{EmoScaleLabels['Arousal']['Labels_En'][i]}\n"
            else: 
                if PromptGenLang is not None:
                    raise ValueError(f"Invalid language: {PromptGenLang}")
                else:
                    raise ValueError(f"Invalid language: {EmoScaleLabels['PromptGenLang']}")
                
    return EmoScalePromptDesc

def REmoScaleDesc(EmoScaleLabels, PromptGenLang) -> RunnableLambda:
    """
    Returns a RunnableLambda that builds a readable description of the emotion scales for the given language.
    """
    return RunnableLambda(BuildReadableEmoScaleDesc(
        EmoScaleLabels=EmoScaleLabels,
        PromptGenLang=PromptGenLang
    ))

output = BuildReadableEmoScaleDesc(EmoScaleLabelsDefs, "Ja")
print(output)