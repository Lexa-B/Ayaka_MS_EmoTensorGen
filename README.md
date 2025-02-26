# Ayaka_MS_EmoTensorGen
_This Microservice is for processing the transient emotional affect of the Ayaka AI's persona as well as the multidirectional estimated emotional state of all other conversational participants, then outputing those into both an LLM-readable format and a contextualized slice of a 5D tensor._

## 1. Overview

There are two main components to this microservice:

1. **The Immidiate Functional Service**: A Python-based LangServe microservice that processes the transient emotional affect of the Ayaka AI's persona as well as the multidirectional estimated emotional state of all other conversational participants. This is intimately tied to the AI chat loop, and is responsible for processing the incoming messages from the user, as well as returning an emotional state context for the AI to use in its response. It supports poly-user conversations, and so can be used in a multi-user chat environment. As such, it is not locked to a hard agent-user-agent-user loop.

2. **The Tensor**: A 5D tensor that contains the emotional state of the Ayaka AI and all other conversational participants over the entirety of the model's existence. Within this service, this tensor is immidiately handled in a sliced-contextualized manner, and and in that form basically serves as a "memory" of the conversation and the ever-changing emotional state of the participants within that one conversation. It is not directly human or LLM-readable, but is readily used to generate the LLM-readable format.
In the tensor's unsliced-uncontextualized form, it cannot be used for generating LLM-readable information, but it offers a mathematical representation of the emotional state of the AI (or rather, the AI's persona as it serves as a homunculus within the conversation) and all other participants over the entirety of the AI's existence. Not only does it show the estimated emotional state of the AI or any user at any point that the AI has interacted with them, but it also shows _directionality_, that is, the emotional states of all participants as directed to all other participants. This allows for complex interconnected webs of emotionality. In it's current form, it's very computationally intensive as it's entirely built through LLM inference.
It is a very sparse tensor, with the majority of the values at any given point being null, with the exception of the emotional state of the AI itself, of course.


### Table of Contents
- [1. Overview](#1-overview)
- [2. The Service Basics](#2-the-service-basics)
    - [a. Overview](#2a-overview)
    - [b. Core Functionalities](#2b-core-functionalities)
    - [c. API Endpoints](#2c-api-endpoints)
    - [d. Error Handling](#2d-error-handling)
    - [e. Performance Optimization](#2e-performance-optimization)
- [3. Architecture](#3-architecture)
- [4. The Service's Code](#4-the-service-code)
- [5. The Tensor](#5-the-tensor)
    - [a. The Tensor's Structure](#5a-the-tensor-structure)
    - [b. The Tensor's Context](#5b-the-tensor-context)
    - [c. The Tensor's Generation](#5c-the-tensor-generation)
- [6. The LLM-Readable Format](#6-the-llm-readable-format)
    - [a. The Format's Structure](#6a-the-format-structure)
    - [b. The Format's Generation](#6b-the-format-generation)

## 5. The Tensor<a name="5-the-tensor"></a>
### The Tensor's Structure<a name="5a-the-tensor-structure"></a>
The tensor is a 5D tensor, with the following dimensions:
5. The Transient Dimension
  This dimension roughly represents the time-frame of the conversation, with each index being a transient or "slice" of the conversation at a given point in time. It does not have a 1-to-1 correspondence to the actual time-frame of the conversation, but rather is a representation of the conversation's progress as it flows through changes in emotion. This could be every message for short messages, or every chunk of coherent emotion for longer messages.
4. The Emoter Dimension
  This dimension represents the frame of reference for the emotional state of the AI's persona or any other participant's inferred emotional state.
3. The Direction Dimension
  This dimension represents the directionality of the emotional state, that is, the emotional state of the AI's persona as directed to any other participant, or the emotional state of any other participant as directed to the AI's persona or any other participant. It forms a complex mesh of emotional connections between all participants.
2. The Emotion Dimension
  This dimension represents the specific emotion family (currently using the Plutchik's Wheel of Emotions model, e.g. joy, trust, sadness, etc., although this can be changed later after further research) that the emotional state is directed towards.
1. The Emotional State Dimension
  This dimension represents the intensity, valence, and arousal of the given emotion family, that is, the strength of the emotion, how much it is positive or negative, and how elivated it's activation-level is (normally this would be a physiological arousal).

### The Tensor's Context<a name="5b-the-tensor-context"></a>
### The Tensor's Generation<a name="5c-the-tensor-generation"></a>