# Python
from typing import List
import os
import re
import ast

# LangChain
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain.schema.runnable.passthrough import RunnableAssign

# Other
import base32_crockford
import dropbox

# Custom
from ayaka_utils.Defs.model_configurator import get_configured_model
from ayaka_utils.Defs.DbxConnector import DbxFIO
from ayaka_utils.Classes.EmoTensorModels import EmoTensorFull_CTXD, EmoTensor4DSlice_CTXD, EmoTensor3DSlice_CTXD, EmoTensor2DSlice_CTXD, EmoTensor1DSlice_CTXD
from Utils.Defs.TensorInit_SlicedContextualized import TensorInit_SlicedContextualized
#from ayaka_utils.Defs.pprint import pprint  

# Custom Runnables
from ayaka_utils.Runnables.RPrint import RPrint
from ayaka_utils.Runnables.RSave import RSave
from ayaka_utils.Runnables.REmoConvHist import GetSimplifiedConversationHistory
from Utils.Runnables.REmoScaleDesc import BuildReadableEmoScaleDesc
from Utils.Defs.EmoDesc_FromXML_ToDict import EmoDesc_FromXML_ToDict

################################################################################
## Local Configs

EmoDescFile = "./Configs/EmoList_Plutchik.xml"

EmoScaleLabelsDefs = ast.literal_eval(open("./Configs/EmoScaleLabels.jsonc", "r").read())

UseDropbox = True

################################################################################
## Model Configuration

model_config_file = "./Configs/EmoTensor.ModelConfig.jsonc"

# Create and configure LLMs
LLM_Scratchpad = get_configured_model(
    model_type="llm_scratchpad", config_file=model_config_file
)
LLM_ScratchSynopsis = get_configured_model(
    model_type="llm_scratch_synopsis", config_file=model_config_file
)
LLM_EmoValues = get_configured_model(
    model_type="llm_emo_values", config_file=model_config_file
)
LLM_EmoValuesFallback = get_configured_model(
    model_type="llm_emo_values_fallback", config_file=model_config_file
)
LLM_EmoContext = get_configured_model(
    model_type="llm_emo_context", config_file=model_config_file
)

# Create and configure embedders
embedder_jp = get_configured_model(
    model_type="embedder_jp", config_file=model_config_file
)
embedder_eng = get_configured_model(
    model_type="embedder_eng", config_file=model_config_file
)

################################################################################
## Get a simplified users list for the LLM to read

def GetSimplifiedUsersList(ConvoUsers: List[dict]) -> str:
    """
    Get a simplified users list for the LLM to read
    """
    def GetGenderString(gender: str) -> str:
        """
        Get a gender string for the LLM to read
        """
        try:
            GenderMap = {
                "woman": "a woman",
                "man": "a man",
                "agender": "an agender person",
                "non-binary": "a non-binary person",
                "other": "an other gender of person"
            }
            return GenderMap[gender]
        except KeyError:
            return "an unknown gender of person"
    
    # Get the list of users
    simplified_users_list = ""
    for user in ConvoUsers:
        UserPronouns = str([value for subdict in ast.literal_eval(user['pronouns']).values() for value in subdict.values()])
        simplified_users_list += (
            f"{user['preferred_name']}:\n"
            f"{user['preferred_name']} ({user['name']} / {user['phonetic_jp']} / {user['phonetic_en']}) "
            f"is {GetGenderString(user['gender'])} who uses the pronouns {UserPronouns} "
            f"and is described as follows:\n「{user['bio']}」\n\n"
        )
    return simplified_users_list

################################################################################
## EmoTensorGen

def TransientEmoScratchGen(input_dict: dict) -> str:
    """
    Generate the scratchpad for the given transient and conversation history, then use it to generate the scratch synopsis.
    """
    TransientEmoterSlice = input_dict["TransientEmoterSlice"]
    SpeakerUid = TransientEmoterSlice["speaker_user"]
    
    # Get the list of emoters
    emoters = TransientEmoterSlice.get("emoters", [])
    # Get the speaker's emoter
    speaker_emoter = next((em for em in emoters if em["emoter_user"] == SpeakerUid), None)
    if speaker_emoter is None:
        raise ValueError("No speaker emoter found in TransientEmoterSlice.")
    
    # Try to get the perceiver's emoter if available
    try:
        perceiver_emoter = next(em for em in emoters if em["emoter_user"] != SpeakerUid)
    except StopIteration:
        perceiver_emoter = None

    # Decide if we're processing the speaker's perspective.
    # If there's only one emoter, assume it's for the speaker.
    is_speaker = True if perceiver_emoter is None else (emoters[0]["emoter_user"] == SpeakerUid)
    
    # Set external contexts; if no perceiver emoter, use an empty string.
    external_context_speaker = speaker_emoter["external_context"]
    external_context_perceiver = perceiver_emoter["external_context"] if perceiver_emoter is not None else ""
    
    # Update TransientEmoterSlice with additional keys so downstream functions can access them.
    TransientEmoterSlice_updated = dict(TransientEmoterSlice)
    TransientEmoterSlice_updated["external_context_speaker"] = external_context_speaker
    TransientEmoterSlice_updated["external_context_perceiver"] = external_context_perceiver
  
    
    ScratchpadPrompt_Speaker = ChatPromptTemplate.from_template(
            "You are an emotional analysis assistant that is part of a larger AI program. "
            "Your job is to use chain-of-thought reasoning to talk through a stream of consciousness "
            "to help think about the emotions the speaker who wrote or said the given message may be feeling right now in this very moment. "
            "What you write here will be used for further analysis, so it's very important to mention everything that's helpful. "
            "Only discuss details that may be relevant to the emotional state of the speaker, "
            "but don't leave out any details that might be important.\n"
            "You will likely be given bilingual Jp/En messages, the language of the message might be relevant if it contains cultural references or idioms, so consider it during analysis if it seems relevant, "
            "but don't let it distract you from the task at hand. "
            "Do not translate any Japanese into English; whenever Japanese is used just reason off of the text as is. "
            "Conversely, if the message is in English and uses a cultural reference or idioms that are only understandable in western culture, please explain the idom in Japanese insofar as it may be relevant to the emotional state of the speaker. "
            "(your analysis will be used by non-native English speakers, so complex English idioms or slang may be misunderstood)\n"
            "When you begin, don't start with any presuppositions about the weight or meaning of the message, "
            "so don't start with pre-analysis like \"This message is important because...\", \"This message is is a complex tapestry of emotions...\", "
            "or \"This simple message carries a lot of emotional weight...\" or anything like that. "
            "Just start by considering the message with steps such as these:\n"
            "1) Quickly reiterate who sent the message, and also who they're likely sending it to if there are multiple people in the conversation.\n"
            "2) Start by listing out everything you know about the speaker ({speaker}) that may be relevant to their emotional state, "
            "including their name, age, gender, location, and any other relevant information. \n"
            "3) Next, list out the same information for anyone else in the conversation as far as it may be pertinent to {speaker}\'s emotional state.\n"
            "4) Then list out everything you know about the histories of the speaker and the other people in the conversation, "
            "being sure to mention the parts of the history that you thought were relevant, and why you thought they were.\n"
            "5) Then, list out everything you know about the current message in the conversation; "
            "think about how words, actions, and emotional subtext of past and current information "
            "might be related to {speaker}\'s emotional state. \n"
            "The speakers in this conversation are as follows:\n"
            "{ConvoUsers_Simplified}\n"
            "The current speaker is: {speaker}\n"
            "The context of the situation is:\n"
            "{EmbedContext}\n"
            "Here is the conversation history, detail will increase as messages become more recent:\n"
            "{conversation_history}\n"
            "The context of {speaker}\'s current message in the conversation is:\n"
            "{external_context_speaker}\n"
            "Here is {speaker}\'s latest message:\n"
            "\"\"\"\n{message}\n\"\"\"\n"
            "Analyze {speaker}\'s latest message. "
            "Think through it step by step. Extra formatting like bullet points or list numbers aren't necessary. "
            "Just talk in a long-form train of thought until anything potentially relevant has been said.\n"
            "** Speak through it in the form of a running stream of conciousness **"
    )
    
    ScratchpadPrompt_Perciever = ChatPromptTemplate.from_template(
            "You are an emotional analysis assistant that is part of a larger AI program. "
            "Your job is to use chain-of-thought reasoning to talk through a stream of consciousness "
            "to help think about the emotions the perciever who received the given message may be feeling right now in this very moment. "
            "What you write here will be used for further analysis, so it's very important to mention everything that's helpful. "
            "Only discuss details that may be relevant to the emotional state of the perciever, "
            "but don't leave out any details that might be important.\n"
            "You will likely be given bilingual Jp/En messages, the language of the message might be relevant if it contains cultural references or idioms, so consider it during analysis if it seems relevant, "
            "but don't let it distract you from the task at hand. "
            "Do not translate any Japanese into English; whenever Japanese is used just reason off of the text as is. "
            "Conversely, if the message is in English and uses a cultural reference or idioms that are only understandable in western culture, please explain the idom in Japanese insofar as it may be relevant to the emotional state of the speaker. "
            "(your analysis will be used by non-native English speakers, so complex English idioms or slang may be misunderstood)\n"
            "When you begin, don't start with any presuppositions about the weight or meaning of the message, "
            "so don't start with pre-analysis like \"This message is important because...\", \"This message is is a complex tapestry of emotions...\", "
            "or \"This simple message carries a lot of emotional weight...\" or anything like that. "
            "Just start by considering the message with steps such as these:\n"
            "1) Quickly reiterate who sent the message, and also who they're likely sending it to if there are multiple people in the conversation.\n"
            "2) Start by listing out everything you know about the perciever ({perciever}) that may be relevant to their emotional state, "
            "including their name, age, gender, location, and any other relevant information. \n"
            "3) Next, list out the same information for the speaker in the conversation, {speaker}, as far as it may be pertinent to {perciever}\'s emotional state.\n"
            "4) Next, also list out the same information for anyone else in the conversation as far as it may be pertinent to {perciever}\'s emotional state.\n"
            "5) Then list out everything you know about the histories of the perciever and the other people in the conversation, "
            "being sure to mention the parts of the history that you thought were relevant, and why you thought they were.\n"
            "6) Then, list out everything you know about the current message in the conversation; "
            "think about how words, actions, and emotional subtext of past and current information "
            "might be related to and affect {perciever}\'s emotional state upon receiving the message. \n"
            "The participants in this conversation are as follows:\n"
            "{ConvoUsers_Simplified}\n"
            "The current perciever is: {perciever}\n"
            "The context of the situation is:\n"
            "{EmbedContext}\n"
            "Here is the conversation history, detail will increase as messages become more recent:\n"
            "{conversation_history}\n"
            "The context of {perciever} when recieving the latest message in the conversation is:\n"
            "{external_context_perceiver}\n"
            "Here is {speaker}\'s latest message:\n"
            "\"\"\"\n{message}\n\"\"\"\n"
            "Analyze {perciever}\'s emotional state upon receiving the message. "
            "Think through it step by step. Extra formatting like bullet points or list numbers aren't necessary. \n"
            "Just talk in a long-form train of thought until anything potentially relevant has been said.\n"
            "** Speak through it in the form of a running stream of conciousness **"
    )
    
    EmoScratchSynopsisPrompt_Speaker = ChatPromptTemplate.from_template(
            "You are an emotional analysis assistant that is part of a larger AI program. "
            "Your job is to evaluate and summarize the emotional synopsis of the specific message\'s speaker at the time of speaking. "
            "Summarize it down into no more that two sentences. "
            "The current speaker is: {speaker}\n"
            "The speakers in this conversation are as follows:\n"
            "{ConvoUsers_Simplified}\n"
            "The context of the situation is:\n"
            "{EmbedContext}\n"
            "Here is the speaker\'s conversation history, detail will increase as messages become more recent:\n"
            "{conversation_history}\n"
            "The context of {speaker}\'s message in the conversation is:\n"
            "{external_context_speaker}\n"
            "Here is {speaker}\'s latest message:\n"
            "\"\"\"\n{message}\n\"\"\"\n"
            "You have already done some cursory analysis of the situation "
            "accompanied by a log of a running stream of thought. "
            "Your thoughts of the situation until now are:\n"
            "\"\"\"\n{scratchpad}\n\"\"\"\n"
            "Now summarize the emotional synopsis of {speaker} at the time of writing this message. "
            "Extra formatting isn't necessary. \n"
            "** State just a one or two-sentence summary, be focused. **"

    )
    
    EmoScratchSynopsisPrompt_Perciever = ChatPromptTemplate.from_template(
            "You are an emotional analysis assistant that is part of a larger AI program. "
            "Your job is to evaluate and summarize the emotional synopsis of the specific message\'s perciever at the time of receiving the message. "
            "The current perciever is: {perciever}\n"
            "The participants in this conversation are as follows:\n"
            "{ConvoUsers_Simplified}\n"
            "The context of the situation is:\n"
            "{EmbedContext}\n"
            "Here is the perciever\'s conversation history, detail will increase as messages become more recent:\n"
            "{conversation_history}\n"
            "The context of the conversation is:\n"
            "{external_context_perceiver}\n"
            "Here is {speaker}\'s latest message:\n"
            "\"\"\"\n{message}\n\"\"\"\n"
            "You have already done some cursory analysis of the situation "
            "accompanied by a log of a running stream of thought. "
            "Your thoughts of the situation until now are:\n"
            "\"\"\"\n{scratchpad}\n\"\"\"\n"
            "Now summarize the emotional synopsis of {perciever} at the time of receiving the message. "
            "Extra formatting isn't necessary. \n"
            "** State just a one or two-sentence summary, be focused. **"
    )

    # Determine if we're processing speaker or perceiver
    is_speaker = TransientEmoterSlice_updated["emoters"][0]["emoter_user"] == TransientEmoterSlice_updated["speaker_user"]

    # Create the base dictionary for the chain, including all required variables.
    base_dict = {
        # The full sliced contextualized tensor file... this is pretty big.
        "TensorFile": input_dict["TensorFile"],
        # A simplified conversation history for the LLM to read, This version is still quite long as it will be used for the scratchpad
        "conversation_history": GetSimplifiedConversationHistory(
            TensorFile=input_dict["TensorFile"],
            PerspectiveUser=input_dict["PerspectiveUser"],
            ConvoUsers=input_dict["ConvoUsers"],
            Fidelity_1=4,
            Fidelity_2=4,
            Fidelity_3=8,
            Fidelity_4=16,
            Fidelity_5=32,
            Fidelity_6=64,
            EmoScaleLabels=EmoScaleLabelsDefs
        ),
        # The user to evaluate from the perspective of
        "PerspectiveUser": input_dict["PerspectiveUser"],
        # The message text
        "message": TransientEmoterSlice["message"],
        # The speaker's preferred name
        "speaker": next(user['preferred_name'] for user in input_dict["ConvoUsers"] if user['id'] == SpeakerUid),
        # The perciever's preferred name
        "perciever": (next(user['preferred_name'] for user in input_dict["ConvoUsers"] if user['id'] == perceiver_emoter["emoter_user"])
                      if perceiver_emoter is not None else ""),
        # The list of conversation users
        "ConvoUsers": input_dict["ConvoUsers"],
        # The list of conversation users, listed in simple LLM-parseable format
        "ConvoUsers_Simplified": GetSimplifiedUsersList(input_dict["ConvoUsers"]),
        # The embed context
        "EmbedContext": input_dict["EmbedContext"],
        # The emotion scale description
        "emo_scale_desc": BuildReadableEmoScaleDesc(EmoScaleLabelsDefs),
        # The TransientEmoterSlice updated with additional keys
        "TransientEmoterSlice": TransientEmoterSlice_updated,
        # The external context of the speaker
        "external_context_speaker": external_context_speaker,
        # The external context of the perciever
        "external_context_perceiver": external_context_perceiver
    }

    # Build the chain: generate scratchpad and scratch synopsis, then merge them into 'TransientEmoterSlice'
    EmoScratchTransient = (
        RunnableLambda(lambda x: base_dict)
        | RunnableAssign({
            "scratchpad": (ScratchpadPrompt_Speaker if is_speaker else ScratchpadPrompt_Perciever)
            #| RPrint(preface=f"Using {'speaker' if is_speaker else 'perciever'} scratchpad prompt: ")
            | RSave(RunningState=(lambda x: x), preface=f"Scratchpad {LLM_Scratchpad}", filename=f"{'speaker' if is_speaker else 'perciever'} scratchpad prompt", path="./Logs/EmoTensorGen/LastPrompts/", source=None, filetype="txt", overwrite=True, suppress_save=False, suppress_print=False, verbose=False)
            #| RPrint(preface=f"Using {'speaker' if is_speaker else 'perciever'} scratchpad prompt: ")
            | LLM_Scratchpad
            | StrOutputParser()
            #| RPrint(preface=f"Generated {'speaker' if is_speaker else 'perciever'} scratchpad: ")
        })
        | RunnableAssign({ # Overwrite conversation history with a more simplified version to save tokens when summarizing.
            "conversation_history": RunnableLambda(lambda x: GetSimplifiedConversationHistory(
                TensorFile=x["TensorFile"],
                PerspectiveUser=x["PerspectiveUser"],
                ConvoUsers=x["ConvoUsers"],
                Fidelity_1=2,
                Fidelity_2=2,
                Fidelity_3=2,
                Fidelity_4=2,
                Fidelity_5=2,
                Fidelity_6=16,
                EmoScaleLabels=EmoScaleLabelsDefs
        )),})
        | RunnableAssign({
            "scratch_synopsis": (EmoScratchSynopsisPrompt_Speaker if is_speaker else EmoScratchSynopsisPrompt_Perciever)
            #| RPrint(preface=f"Using {'speaker' if is_speaker else 'perciever'} scratch synopsis prompt: ")
            | RSave(RunningState=(lambda x: x), preface=f"Scratchpad {LLM_ScratchSynopsis}", filename=f"{'speaker' if is_speaker else 'perciever'} scratch synopsis prompt", path="./Logs/EmoTensorGen/LastPrompts/", source=None, filetype="txt", overwrite=True, suppress_save=False, suppress_print=False, verbose=False)
            | LLM_ScratchSynopsis
            | StrOutputParser()
            #| RPrint(preface=f"Generated {'speaker' if is_speaker else 'perciever'} scratch synopsis: ")
        })
        | RunnableAssign({
            "TransientEmoterSlice": RunnableLambda(lambda d: {
                **d["TransientEmoterSlice"],
                "scratchpad": d["scratchpad"],
                "scratch_synopsis": d["scratch_synopsis"]
            })
        })
    )

    return EmoScratchTransient

def clean_json_string(message) -> str:
    """Clean up JSON string before parsing"""
    # Get content from LangChain message object
    text = message.content if hasattr(message, 'content') else str(message)
    # Replace smart quotes with straight quotes
    text = text.replace('"', '"').replace('"', '"')
    # Remove any extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def GenerateEmoValues(input_dict: dict, target_emotion: str, GenLang: str = "En") -> EmoTensor1DSlice_CTXD:
    parser = PydanticOutputParser(pydantic_object=EmoTensor1DSlice_CTXD)

    # Get the emotion description from the EmoDesc_FromXML_ToDict function
    emotion_description = next(item['EmoDesc'] for item in EmoDesc_FromXML_ToDict(EmoDescFile, Lang=GenLang) if item['EmoFamily'] == target_emotion)

    # Get the schema for the JSON format
    schema = parser.get_format_instructions()
    
    # Determine if we're processing speaker or perceiver by checking the emoter user
    is_speaker = input_dict["TransientEmoterSlice"]["emoters"][0]["emoter_user"] == input_dict["TransientEmoterSlice"]["speaker_user"]
    
    GenerateEmoValuesPrompt_Speaker = ChatPromptTemplate.from_template(
        "You are an emotional analysis assistant that is part of a larger AI program. "
        "Your job is to evaluate one specific emotional component of {speaker}'s emotional state at the exact moment of speaking. "
        "Base your analysis in the Plutchik Emotion Wheel theory of emotions. "
        "The emotion you are evaluating is: {emotion}\n"
        "Here is the description of the emotion for the purposes of this evaluation:\n"
        "{emotion_description}\n"
        "The metrics used to evaluate the intensity, valence, and arousal of emotions are as follows:\n"
        "{emo_scale_desc}\n"
        "Based on everything you've learned about {speaker}, the other people in the conversation, "
        "the message history, etc., you'll have to make an educated deduction as to "
        "the intensity, valence, and arousal of the emotion that they are feeling at this exact moment. "
        "Although their emotional state is informed by the past, only evaluate for the emotion as it is felt at this exact moment, "
        "therefore the values you select will be almost entirely based on the current message, followed by the past few messages. "
        "The current speaker is: {speaker}\n"
        "The participants in this conversation are as follows:\n"
        "{ConvoUsers_Simplified}\n"
        "The context of the situation is:\n"
        "{EmbedContext}\n"
        "Here is the speaker's conversation history, detail will increase as messages become more recent:\n"
        "{conversation_history}\n"
        "Here is the speaker's message:\n"
        "\"\"\"\n{message}\n\"\"\"\n"
        "Here is the running stream-of-thought analysis of the speaker's emotional state:\n"
        "\"\"\"\n{scratchpad}\n\"\"\"\n"
        "Here is the speaker's emotional synopsis:\n"
        "\"\"\"\n{scratch_synopsis}\n\"\"\"\n"
        "Again, only evaluate for the Plutchik Emotion Wheel emotion: {emotion}, and only evaluate the emotion as it is felt by the speaker, {speaker}. "
        "Not all text you are given will be relevant to the emotion you are evaluating. "
        "Often, you will be given a message that is very low intensity in the emotion you are evaluating. "
        "It is okay to assign low numerical values for emotions in that case. "
        "Remember to scale the intensity, valence, and arousal of the emotion based on the \"Intensity Scale\" curve provided above, "
        "such as logarithmic, sigmoid, or logistic. Because of this, values will still very likely be above 0.00 "
        "even if the apparent expression of the emotion is quite low. "
        "Answer with two decimal places of precision. "
        "Provide your answer in this JSON format:\n"
        "{{\n"
        "\t\"emotion\":\"{emotion}\",\n"
        "\t\"context\":\"<String describing the context of this particular emotion at this exact point in time. Be brief and focused. It's not necessary to restate the names of the people involved, just the context of the emotion.>\",\n"
        "\t\"intensity\":<float between 0.00 and 1.00>,\n"
        "\t\"valence\":<float between -1.00 and 1.00>,\n"
        "\t\"arousal\":<float between 0.00 and 1.00>\n"
        "}}"
    )
    
    GenerateEmoValuesPrompt_Perciever = ChatPromptTemplate.from_template(
        "You are an emotional analysis assistant that is part of a larger AI program. "
        "Your job is to evaluate one specific emotional component of {perciever}'s emotional state at the exact moment of receiving the message from {speaker} in the context of the message they just said. "
        "Base your analysis in the Plutchik Emotion Wheel theory of emotions. "
        "The emotion you are evaluating is: {emotion}\n"
        "Here is the description of the emotion for the purposes of this evaluation:\n"
        "{emotion_description}\n"
        "The metrics used to evaluate the intensity, valence, and arousal of emotions are as follows:\n"
        "{emo_scale_desc}\n"
        "Based on everything you've learned about {perciever}, the other people in the conversation, "
        "the message history, etc., you'll have to make an educated deduction as to "
        "the intensity, valence, and arousal of the emotion that they are feeling at this exact moment. "
        "Although their emotional state is informed by the past, only evaluate for the emotion as it is felt at this exact moment, "
        "therefore the values you select will be almost entirely based on the current message, followed by the past few messages. "
        "The current speaker is: {speaker}\n"
        "The participants in this conversation are as follows:\n"
        "{ConvoUsers_Simplified}\n"
        "The context of the situation is:\n"
        "{EmbedContext}\n"
        "Here is the conversation history, detail will increase as messages become more recent:\n"
        "{conversation_history}\n"
        "Here is the speaker's message:\n"
        "\"\"\"\n{message}\n\"\"\"\n"
        "Here is the running stream-of-thought analysis of the perciever's emotional state:\n"
        "\"\"\"\n{scratchpad}\n\"\"\"\n"
        "Here is the perciever's emotional synopsis:\n"
        "\"\"\"\n{scratch_synopsis}\n\"\"\"\n"
        "Again, only evaluate for the Plutchik Emotion Wheel emotion: {emotion}, and only evaluate the emotion as it is felt by the perciever, {perciever} in the context of the message they just received from {speaker}. "
        "Not all text you are given will be relevant to the emotion you are evaluating. "
        "Often, you will be given a message that is very low intensity in the emotion you are evaluating. "
        "It is okay to assign low numerical values for emotions in that case. "
        "Remember to scale the intensity, valence, and arousal of the emotion based on the \"Intensity Scale\" curve provided above, "
        "such as logarithmic, sigmoid, or logistic. Because of this, values will still very likely be above 0.00 "
        "even if the apparent expression of the emotion is quite low. "
        "Answer with two decimal places of precision. "
        "Provide your answer in this JSON format:\n"
        "{{\n"
        "\t\"emotion\":\"{emotion}\",\n"
        "\t\"context\":\"<String describing the context of this particular emotion at this exact point in time. Be brief and focused. It's not necessary to restate the names of the people involved, just the context of the emotion.>\",\n"
        "\t\"intensity\":<float between 0.00 and 1.00>,\n"
        "\t\"valence\":<float between -1.00 and 1.00>,\n"
        "\t\"arousal\":<float between 0.00 and 1.00>\n"
        "}}"
    )
    
    fallback_prompt = ChatPromptTemplate.from_template(
        "This is the output from another model that failed to parse. "
        "Ensure that the output exactly matches the JSON format provided. "
        "The output should appear as follows:\n"
        "{{\n"
        "\t\"emotion\":\"{emotion}\",\n"
        "\t\"context\":\"<string describing the context of this particular emotion. No more than two sentences, but one is better. Doesn't need to restate names of people, just the context of the emotion.>\",\n"
        "\t\"intensity\":<float between 0.00 and 1.00>,\n"
        "\t\"valence\":<float between -1.00 and 1.00>,\n"
        "\t\"arousal\":<float between 0.00 and 1.00>\n"
        "}}\n\n"
        "Here is the schema for the JSON format:\n"
        "{schema}"
        "Your job is to fix the input to match the required format. "
        "Do not explain the mistake. "
        "Do not say anything else. Just output the fixed JSON."
    )
    
    def can_parse(output: str) -> bool:
        try:
            # Try to parse the output.
            parser.parse(output)
            return True
        except Exception:
            return False
    
    GenerateEmoValuesChain = (
        RunnableLambda(lambda x: {
            "PerspectiveUser": input_dict["PerspectiveUser"],
            "message": input_dict["TransientEmoterSlice"]["message"],
            "speaker": input_dict["speaker"],
            "perciever": input_dict["perciever"],
            # Just use the simplified list of conversation users from the last chain
            "ConvoUsers_Simplified": input_dict["ConvoUsers_Simplified"],
            "EmbedContext": input_dict["EmbedContext"],
            "external_context_speaker": input_dict["TransientEmoterSlice"]["external_context_speaker"],
            "external_context_perceiver": input_dict["TransientEmoterSlice"]["external_context_perceiver"],
            "scratchpad": input_dict["TransientEmoterSlice"]["scratchpad"],
            "scratch_synopsis": input_dict["TransientEmoterSlice"]["scratch_synopsis"],
            "conversation_history": GetSimplifiedConversationHistory(
                TensorFile=input_dict["TensorFile"],
                PerspectiveUser=input_dict["PerspectiveUser"],
                ConvoUsers=input_dict["ConvoUsers"],
                Fidelity_1=2,
                Fidelity_2=2,
                Fidelity_3=2,
                Fidelity_4=2,
                Fidelity_5=2,
                Fidelity_6=16,
                EmoScaleLabels=EmoScaleLabelsDefs
            ),
            "emotion": target_emotion,
            "emo_scale_desc": input_dict["emo_scale_desc"],
            "emotion_description": emotion_description
        })
        #| RPrint(preface="Passing to emotion values prompt: ")
        | (GenerateEmoValuesPrompt_Speaker if is_speaker else GenerateEmoValuesPrompt_Perciever)
        #| RPrint(preface=f"Using {'speaker' if is_speaker else 'perceiver'} {'emotion'} prompt: ")
        #| RPrint(preface="Generated emotion values prompt: ")
        | RSave(RunningState=(lambda x: x), preface=f"EmoValues {LLM_EmoValues} for {target_emotion}", filename=f"{'speaker' if is_speaker else 'perciever'} emo values prompt - {target_emotion}", path="./Logs/EmoTensorGen/LastPrompts/", source=None, filetype="txt", overwrite=True, suppress_save=False, suppress_print=False, verbose=False)
        | LLM_EmoValues
        | RunnableLambda(clean_json_string)
        #| RPrint(preface="Reasoning complete: ")
        | RunnableBranch(
            (lambda x: can_parse(x), RunnableLambda(lambda x: x) 
                | parser),
            lambda x: RunnableLambda(lambda x: x) 
                | RPrint(preface="Failed to parse: ")
                | StrOutputParser() 
                | RunnableLambda(lambda x: {"input": x, "schema": schema, "emotion": target_emotion})
                | fallback_prompt
                | RSave(RunningState=(lambda x: x), preface=f"EmoValues fallback {LLM_EmoValuesFallback} for {target_emotion}", filename=f"Fallback emo values prompt - {target_emotion}", path="./Logs/EmoTensorGen/LastPrompts/", source=None, filetype="txt", overwrite=True, suppress_save=False, suppress_print=False, verbose=False)
                | LLM_EmoValuesFallback
                | parser
        )
        #| RPrint(preface="Output parsed: ")
    )
    
    return GenerateEmoValuesChain


################################################################################
## Generate the emotional context output of the current transient

def GenEmoContext(input_dict: dict):

    EmoContextPrompt = ChatPromptTemplate.from_template( # ToDo: Make this a more general prompt that can be used for any AI
        "Rewrite this psychonanlytical report into the first person from {Emoter}'s perspective. Don't materially change the content of the message except for cases that don't make sense in the 1st-person, such as:\n"
        " * Change words like 'likely' & 'probably' if they don't fit to more appropriate 1st-person experiential words\n"
        " * If the message has any terminology like \"analyzing {Emoter}'s message\" \"Evaluating {Emoter}'s feelings about\" change those into an experiential present form that fluidly makes sense, or remove it entirely\n"
        " * Also change or remove any language that sounds overly analytical or too distant... make it personal and explanational.\n"
        " * {Emoter} uses the pronouns '{FirstPerPronoun}'.\n"
        "It should be smooth and natural. "
        "Just make it sound like {Emoter} is thinking it in the present:\n"
        "\"\"\"\n"
        "\"{Scratchpad}\"\n"
        "Here is the speaker's emotional synopsis: \"{ScratchSynopsis}\""
    )

    GenEmoContextChain = (
        RunnableLambda(lambda x: {
            "Emoter": input_dict["Emoter"],
            "FirstPerPronoun": input_dict["FirstPerPronoun"],
            "Scratchpad": input_dict["Scratchpad"],
            "ScratchSynopsis": input_dict["ScratchSynopsis"]
            })
        | EmoContextPrompt
        #| RPrint(preface="EmoContext prompt: ")
        | RSave(RunningState=(lambda x: x), preface=f"EmoContext {LLM_EmoContext}", filename="EmoContext prompt", path="./Logs/EmoTensorGen/LastPrompts/", source=None, filetype="txt", overwrite=True, suppress_save=False, suppress_print=False, verbose=False)
        | LLM_EmoContext
        | StrOutputParser()
        | RPrint(preface="EmoContext generated: ")
    )

    return GenEmoContextChain.invoke(input_dict)

################################################################################
## Main LangChain Chain

EmoTensorChain = (
    TransientEmoScratchGen
    | RunnableAssign({"ThisEmoTensor1DSlice":
                      RunnableParallel(
                            {
                                "Joy": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Joy")),
                                "Trust": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Trust")),
                                "Fear": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Fear")),
                                "Surprise": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Surprise")),
                                "Sadness": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Sadness")),
                                "Disgust": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Disgust")),
                                "Anger": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Anger")),
                                "Anticipation": RunnableLambda(lambda x: GenerateEmoValues(x, target_emotion="Anticipation")),
                            }
                        )
                      })
    #| RPrint(preface="EmoTensorChain complete: ")
)

################################################################################
## Evaluate the current transient

def ProcessEmotionalTransient(
        PerspectiveUser: str,
        ThisMessage: str, 
        SpeakerUser: str, 
        Emoters: List[dict], 
        EmbedContext: str, 
        EmoTensorContextualizedFilePath: str, 
        ConversationUsers: List[dict],
        TestMode: bool = False):
    """
    This function processes an emotional transient and updates the EmoTensorContextualizedFile.

    Args:
        ThisMessage: The message to process.
        SpeakerUser: The user who is speaking.
        Emoters: The emoters in the conversation.
        EmbedContext: The contextual information about the conversation.
        EmoTensorContextualizedFilePath: The file to update with the new transient.
        ConversationUsers: The users in the conversation.
    """

    CurrentEmoTensor4DSliceInput = {
        "message": ThisMessage,
        "speaker_user": SpeakerUser,
        "emoters": Emoters,
    }

    ## Load in the history from the file unless TestMode is True
    if not TestMode:
        if UseDropbox:
            # Read the file from Dropbox
            try:
                json_data_in = DbxFIO(DbxPath=EmoTensorContextualizedFilePath, mode="read", type="json")
            except dropbox.exceptions.ApiError as e:
                if (isinstance(e.error, dropbox.files.DownloadError) and 
                    e.error.is_path() and 
                    e.error.get_path().is_not_found()):
                    # Handle case where file doesn't exist - create new tensor
                    print("File not found in Dropbox, creating new tensor")
                    EmoTensorFull = TensorInit_SlicedContextualized(ConversationUsers, EmoDescFile)
                else:
                    # Re-raise other Dropbox API errors
                    print(f"Dropbox API error: {e}")
                    raise e
        else:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(EmoTensorContextualizedFilePath), exist_ok=True)
            with open(EmoTensorContextualizedFilePath) as f:
                json_data_in = f.read()

            EmoTensorFull = EmoTensorFull_CTXD.model_validate_json(json_data_in)

    ## If TestMode is True, create a new empty EmoTensorFull object
    else:        
        EmoTensorFull = TensorInit_SlicedContextualized(ConversationUsers, EmoDescFile)

    # ToDo: Check if the EmoTensorFull object is all correct.
    #  - order_1_attributes:
    #    - Check if the attributes are the same as the measures in the EmoList_*.xml and EmoScaleLabels.jsonc
    #  - order_2_emotions: 
    #    - Check if the emotions are the same as the measures in the EmoList_*.xml
    #  - order_3_target:
    #    - Check if the targets are the same as the users in the ConversationUsers list and the order is correct.
    #  - order_4_emoters: 
    #    - Check if the emoters are the same as the users in the ConversationUsers list and the order is correct.
    #  - order_5_transients: 
    #    - Check if the transients' first number is always the same, corresponds to a chat conversation DB, and maintain Crockford Base32 encoding.
    #    - Check if the transients' second number is always the same, corresponds to a chat ID, and maintain Crockford Base32 encoding.
    #    - Check if the transients' final number are in order, don't restart, and maintain Crockford Base32 encoding.
    #  - user_prefnames: 
    #    - Check if the user_prefnames are the same as the users in the ConversationUsers list.
    #  - transients: 
    #    - Check if there are as many transients as there are in the order_5_transients in the EmoTensorFull object.

    # Initialize list to store all processed emoters in order by user ID
    ConvUidsList = [user["id"] for user in ConversationUsers]
    
    ConvUidsList_UserOrder = sorted(ConvUidsList, key=lambda x: int(x.split("=")[1]))

    # Next, pull the speaker to the front of the list
    SpeakerUid = next(
        em['emoter_user'] for em in CurrentEmoTensor4DSliceInput["emoters"]
        if em["emoter_user"] == CurrentEmoTensor4DSliceInput["speaker_user"]
    )
    ConvUidsList_SpeakerOrder = [SpeakerUid] + [x for x in ConvUidsList_UserOrder if x != SpeakerUid]

    Emoters_SpeakerOrder = sorted(CurrentEmoTensor4DSliceInput["emoters"], key=lambda d: ConvUidsList_SpeakerOrder.index(d['emoter_user']))

    ################################################################################
    ## Processing the transient for saving

    ProcessedEmoters = []

    for i, emoter in enumerate(Emoters_SpeakerOrder):

        # Build a slice of the transient for the current emoter
        ThisTransientEmoterSlice = {
            # The message is the same for all emoters
            "message": CurrentEmoTensor4DSliceInput["message"],
            # The speaker is the same for all emoters
            "speaker_user": CurrentEmoTensor4DSliceInput["speaker_user"],
            # The emoters are the current emoter (alone if the speaker), or the current emoter and the speaker (if this emoter is a perceiver)
            "emoters": [emoter] if i == 0 else [emoter, next(em for em in Emoters_SpeakerOrder if em["emoter_user"] == CurrentEmoTensor4DSliceInput["speaker_user"])]
        }

        # Invoke the chain
        chain_result = EmoTensorChain.invoke({
            "TransientEmoterSlice": ThisTransientEmoterSlice,
            "TensorFile": EmoTensorFull,
            "EmbedContext": EmbedContext,
            "PerspectiveUser": PerspectiveUser, # Don't confuse this with the current emoter; it's the perspective user the AI is taking for overall tensor generation.
            "ConvoUsers": ConversationUsers
        })

        processed_emoter = EmoTensor3DSlice_CTXD(
            emoter_user=emoter["emoter_user"],
            external_context=emoter["external_context"],
            targets=[
                EmoTensor2DSlice_CTXD(
                    this_target=emoter["targets"][0]["this_target"],
                    scratch_context=chain_result['scratchpad'],
                    scratch_synopsis=chain_result['scratch_synopsis'],
                    emotions=list(chain_result['ThisEmoTensor1DSlice'].values())
                )
            ]
        )

        ProcessedEmoters.append(processed_emoter)

    # Sort the ProcessedEmoters back into Uid order.
    ProcessedEmoters = sorted(ProcessedEmoters, key=lambda x: ConvUidsList_UserOrder.index(x.emoter_user))

    # Build the final transient for saving (now in file order)
    NewEmoTensor4DSlice = EmoTensor4DSlice_CTXD(
        message=CurrentEmoTensor4DSliceInput["message"],
        speaker_user=CurrentEmoTensor4DSliceInput["speaker_user"],
        emotional_context=GenEmoContext({
            "Emoter": next(d for d in ConversationUsers if d["id"]==ProcessedEmoters[0].emoter_user)["preferred_name"],
            "FirstPerPronoun": ast.literal_eval(next(d for d in ConversationUsers if d["id"]==ProcessedEmoters[0].emoter_user)["pronouns"])["Jp"]["fpp"],
            "Scratchpad": ProcessedEmoters[0].targets[0].scratch_context, # Get the scratchpad from the first emoter (normally Ayaka)
            "ScratchSynopsis": ProcessedEmoters[0].targets[0].scratch_synopsis # Get the scratch synopsis from the first emoter (normally Ayaka)
        }),
        emoters=ProcessedEmoters
    )

    # Increment order_5_transients only once.
    # ToDo: get this from the DB
    if len(EmoTensorFull.order_5_transients) == 0:
        next_transient = "c_0_0_0" # If there are no transients, start at 0
    else:
        last_transient = EmoTensorFull.order_5_transients[-1]
        last_segment = last_transient.split('_')[-1]
        try:
            last_number = base32_crockford.decode(last_segment)
            next_number = last_number + 1
            next_transient = f"c_0_0_{base32_crockford.encode(next_number)}"
        except ValueError:
            # If we can't decode the last segment, start a new sequence
            next_transient = "c_0_0_0"
    EmoTensorFull.order_5_transients.append(next_transient)

    # Append the new transient to the conversation history.
    EmoTensorFull.transients.append(NewEmoTensor4DSlice)

    if UseDropbox:
        DbxFIO(DbxPath=EmoTensorContextualizedFilePath, mode="write", type="json", data=EmoTensorFull.model_dump_json(indent=2))
    else:
        with open(EmoTensorContextualizedFilePath, "w") as f:
            f.write(EmoTensorFull.model_dump_json(indent=2))

    # # Overwrite the message history file.
    # with open(EmoTensorContextualizedFile, "w") as f:
    #     f.write(EmoTensorFull.model_dump_json(indent=2))

    return True