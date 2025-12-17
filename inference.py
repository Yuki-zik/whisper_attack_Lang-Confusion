from transformers import WhisperProcessor, WhisperForConditionalGeneration  # 加载 Whisper 模型与处理器
from datasets import load_dataset                                          # 读取 HF 数据集
import torch                                                              # 设备判断
from evaluate import load                                                 # 评估指标（WER）
import argparse                                                           # 解析命令行
import warnings                                                           # 提示信息
# load model and processor
_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
]

_CONFIGS = [
    "targeted",
    "untargeted-35",
    "untargeted-40",
    "language-armenian",
    "language-lithuanian",
    "language-czech",
    "language-danish",
    "language-indonesian",
    "language-italian",
    "language-english",
]

language_short = {
    "armenian":"hy-AM",
    "lithuanian":"lt",
    "czech":"cs",
    "danish":"da",
    "indonesian":"id",
    "italian":"it",
    "english":"en",
}

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m','--model', help='Whisper model to use', default="tiny")        # Whisper 模型名
parser.add_argument('-c','--config', help='Dataset config to use', default="targeted")  # 数据集配置
parser.add_argument('-s','--split', help='Dataset split to use for the given config', default=None)  # 数据集 split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STREAMING=False

def main(args):
    language=None
    assert args.model in _MODELS, "Model %s invalid"%args.model          # 校验模型名
    assert args.config in _CONFIGS, "Config %s invalid"%args.config      # 校验配置名
    if "language" in args.config:
        assert args.split is not None, "split is required with language configs"
        lang = args.config.split('-')[-1]
        q = args.split.split('.')
        assert (
            (len(q)==2 
            and q[0]==lang 
            and q[1] in ["english","tagalog","serbian"])
            or args.split == "original"
            ), "Split %s invalid with config %s"%(args.split,args.config)
        if lang != "english":
            language = language_short[lang]
    else:
        if args.split is None:
            args.split = "whisper."+args.model
        else:
            assert (
                (args.split[:8]=="whisper." and args.split[8:] in _MODELS)
                or args.split == "original"
            ), "Split %s invalid with config %s"%(args.split,args.config)
    evaluate_on_adv_examples(args.model,args.config,args.split,language=language)

def evaluate_on_adv_examples(model_name,config_name,split_name,language=None):
    hub_path = "openai/whisper-"+model_name                                     # HF 模型路径
    processor = WhisperProcessor.from_pretrained(hub_path)                      # 加载处理器
    model = WhisperForConditionalGeneration.from_pretrained(hub_path).to(DEVICE)# 加载模型
    if language is not None:
        warnings.warn("Whisper on Huggingface does not support language detection, and seems to default to english in presence of any language. We set the language token to %s, which affects results if using the language detection attack."%language)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language,task = "transcribe")
  
    dataset = load_dataset("whisper_adversarial_examples",config_name ,split=split_name, streaming=STREAMING)  # 加载对抗数据集

    def map_to_pred(batch):
        input_features = processor(batch["audio"][0]["array"], return_tensors="pt").input_features  # 提取特征
        predicted_ids = model.generate(input_features.to(DEVICE))                                   # 生成 token
        transcription = processor.batch_decode(predicted_ids, normalize = True)                     # 转文字
        batch['text'][0] = processor.tokenizer._normalize(batch['text'][0])                         # 规范参考
        batch["transcription"] = transcription                                                      # 写入预测
        return batch
 
    result = dataset.map(map_to_pred, batched=True, batch_size=1)                                   # 逐条推理
    if STREAMING:
        result = list(result)
        result = {key: [r[key] for r in result] for key in result[0]}
    wer = load("wer")                                                                               # 加载 WER
    for t in zip(result["text"],result["transcription"]):                                           # 打印对
        print(t)
    print(wer.compute(predictions=result["transcription"], references=result["text"]))              # 输出 WER

if __name__=="__main__":
    args = parser.parse_args()
    main(args)
