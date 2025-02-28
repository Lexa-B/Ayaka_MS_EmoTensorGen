# Python
from lxml import etree

################################################################################
## Import Emotion Descriptions from XML and strip whitespace/newlines, 
## then format and store in EmoInfoList

def EmoDesc_FromXML_ToDict(EmoDescFile, Lang="Ja"):
    def strip_texts(elem):
        if elem.text:
            elem.text = elem.text.strip()
        for child in elem:
            if child.tail:
                child.tail = child.tail.strip()
            strip_texts(child)

    Parser_XML = etree.XMLParser(remove_blank_text=True)
    EmoDesc_FromXML = etree.parse(EmoDescFile, Parser_XML)
    EmoDesc_FromXML_root = EmoDesc_FromXML.getroot()

    strip_texts(EmoDesc_FromXML_root)

    ## First, find the supported languages, and make sure it at least contains the languages supported by this version of this script
    LangSupport_Local = ["Ja", "En"] # Supported languages by this version of this script, Title case only
    LangSupport_File = []
    for child in EmoDesc_FromXML_root:
        if child.tag == "Languages":
            for subchild in child:
                if subchild.tag == "Language":
                    LangSupport_File.append(subchild.attrib.get("Lang").title()) # Convert to title case, then append to list

    for Lang_Local in LangSupport_Local:
        if Lang_Local not in LangSupport_File:
            raise ValueError(f"Language {Lang_Local} is not supported by the file {EmoDescFile}")

    #First, get the measures
    EmoMesures = []
    for child in EmoDesc_FromXML_root:
        if child.tag == "EvaluationMeasures":
            for subchild in child:
                if subchild.tag == "Measure":
                    for sub2child in subchild:
                        if sub2child.tag == "MeasureName":
                            if (Lang == "Ja" and sub2child.attrib.get("Lang") == "Ja"):
                                Metric = sub2child.text
                            elif Lang == "En" and sub2child.attrib.get("Lang") == "En":
                                Metric = sub2child.text

                    EmoMesures.append(Metric)

    #Then, get the emotion families
    EmoInfoList = []
    for child in EmoDesc_FromXML_root:
        if child.tag == "ListOfEmotionFamilies":
            # Emotion Family Loop
            for subchild in child:
                if subchild.tag == "EmotionFamily":
                    if Lang == "Ja":
                        EmotionOfInterest = subchild.attrib.get("Name-Ja")
                    elif Lang == "En":
                        EmotionOfInterest = subchild.attrib.get("Name-En")
                    for subsubchild in subchild:
                        EmoMeasures = [] # Reset the list of measures for each emotion family
                        i = 0 # Reset the index for the measure list lookup
                        if subsubchild.tag == "Definition":
                            if (Lang == "Ja" and subsubchild.attrib.get("Lang")) == "Ja":
                                EmoDef = subsubchild.text
                            elif Lang == "En" and subsubchild.attrib.get("Lang") == "En":
                                EmoDef = subsubchild.text
                        elif subsubchild.tag == "Measures":
                            for sub3child in subsubchild:
                                if (Lang == "Ja" and sub3child.attrib.get("Lang")) == "Ja":
                                    EmoMeasures.append({"Measure": EmoMesures[i], "Desc": sub3child.text})
                                    i += 1
                                elif Lang == "En" and sub3child.attrib.get("Lang") == "En":
                                    EmoMeasures.append({"Measure": EmoMesures[i], "Desc": sub3child.text})
                                    i += 1
                    if not EmoDef: # If no definition is found, raise an error
                        raise ValueError(f"No definition found for {EmotionOfInterest}")

                    EmoFamily = EmotionOfInterest
        
                    if Lang == "Ja":
                        EmoDesc = (
                            f"感情の説明 - {EmotionOfInterest}:\n"
                            f"* 定義: {EmoDef}\n"
                        )
                    elif Lang == "En":
                        EmoDesc = (
                            f"Emotion Description - {EmotionOfInterest}:\n"
                            f"* Definition: {EmoDef}\n"
                        )
                    
                    EmoInfoList.append({
                        "EmoFamily": EmoFamily,
                        "EmoDesc": EmoDesc,
                        "EmoMeasures": EmoMeasures
                    })

    return EmoInfoList