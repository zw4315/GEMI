#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import os, json, re, time

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import ollama
import soundfile as sf
from kokoro_onnx import Kokoro

#监听
import pyaudio
import wave
import numpy as np

from utils.my_log import logger
from utils.common import Common
from utils.config import Config
from utils.my_handle import My_handle

import traceback

config_path = "config.json"
my_handle = None
common = None
config = None

def audio_listen(volume_threshold = 800.0, silence_threshold = 15):
    audio = pyaudio.PyAudio()

    FORMAT = pyaudio.paInt16
    CHANNELS = config.get("talk", "CHANNELS")
    RATE = config.get("talk", "RATE")
    CHUNK = 1024

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=int(config.get("talk", "device_index")),
    )

    frames = []  # 存储录制的音频帧

    is_speaking = False  # 是否在说话
    silent_count = 0  # 沉默计数

    print("[即将开始录音……]")

    while True:
        # 播放中不录音
        if config.get("talk", "no_recording_during_playback"):
            # 存在待合成音频 或 已合成音频还未播放 或 播放中 或 在数据处理中
            if (
                my_handle.is_audio_queue_empty() != 15
                or my_handle.is_handle_empty() == 1
                or wait_play_audio_num > 0
            ):
                time.sleep(
                    float(
                        config.get(
                            "talk", "no_recording_during_playback_sleep_interval"
                        )
                    )
                )
                continue

        # 读取音频数据
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.short)
        max_dB = np.max(audio_data)

        if max_dB > volume_threshold:
            is_speaking = True
            silent_count = 0
        elif is_speaking is True:
            silent_count += 1

        if is_speaking is True:
            frames.append(data)

        if silent_count >= silence_threshold:
            break

    print("[语音录入完成]")

    return frames

# 处理聊天逻辑 传入ASR后的文本内容
def talk_handle(content: str):
    global is_talk_awake

    def clear_queue_and_stop_audio_play(message_queue: bool=True, voice_tmp_path_queue: bool=True, stop_audio_play: bool=True):
        """
        清空队列 或 停止播放音频
        """
        if message_queue:
            ret = my_handle.clear_queue("message_queue")
            if ret:
                logger.info("清空待合成消息队列成功！")
            else:
                logger.error("清空待合成消息队列失败！")
        if voice_tmp_path_queue:
            ret = my_handle.clear_queue("voice_tmp_path_queue")
            if ret:
                logger.info("清空待播放音频队列成功！")
            else:
                logger.error("清空待播放音频队列失败！")
        if stop_audio_play:
            ret = my_handle.stop_audio("pygame", True, True)

    try:
        # 检查并切换聊天唤醒状态
        def check_talk_awake(content: str):
            """检查并切换聊天唤醒状态

            Args:
                content (str): 聊天内容

            Returns:
                dict:
                    ret 是否需要触发
                    is_talk_awake 当前唤醒状态
                    first 是否是第一次触发 唤醒or睡眠，用于触发首次切换时的特殊提示语
            """
            global is_talk_awake

            # 判断是否启动了 唤醒词功能
            # 判断现在是否是唤醒状态
            if is_talk_awake is False:
                # 判断文本内容是否包含唤醒词
                trigger_word = common.find_substring_in_list(
                    content, config.get("talk", "wakeup_sleep", "wakeup_word")
                )
                if trigger_word:
                    is_talk_awake = True
                    logger.info("[聊天唤醒成功]")
                    return {
                        "ret": 0,
                        "is_talk_awake": is_talk_awake,
                        "first": True,
                        "trigger_word": trigger_word,
                    }
                return {
                    "ret": -1,
                    "is_talk_awake": is_talk_awake,
                    "first": False,
                }
            else:
                # 判断文本内容是否包含睡眠词
                trigger_word = common.find_substring_in_list(
                    content, config.get("talk", "wakeup_sleep", "sleep_word")
                )
                if trigger_word:
                    is_talk_awake = False
                    logger.info("[聊天睡眠成功]")
                    return {
                        "ret": 0,
                        "is_talk_awake": is_talk_awake,
                        "first": True,
                        "trigger_word": trigger_word,
                    }
                return {
                    "ret": 0,
                    "is_talk_awake": is_talk_awake,
                    "first": False,
                }
            return {"ret": 0, "is_talk_awake": True, "trigger_word": "", "first": False}

        # 输出识别结果
        logger.info("识别结果：" + content)

        # 空内容过滤
        if content == "":
            return

        username = config.get("talk", "username")

        data = {"platform": "本地聊天", "username": username, "content": content}
        
        # 检查并切换聊天唤醒状态
        check_resp = check_talk_awake(content)
        if check_resp["ret"] == 0:
            # 唤醒情况下
            if check_resp["is_talk_awake"]:
                # 长期唤醒、且不是首次触发的情况下，后面的内容不会携带触发词，即使携带了也不应该进行替换操作
                if config.get("talk", "wakeup_sleep", "mode") == "长期唤醒" and not check_resp["first"]:
                    pass
                else:
                    # 替换触发词为空
                    content = content.replace(check_resp["trigger_word"], "").strip()

                # 因为唤醒可能会有仅唤醒词的情况，所以可能出现首次唤醒，唤醒词被过滤，content为空清空，导致不播放唤醒提示语，需要处理
                if content == "" and not check_resp["first"]:
                    return
                
                # 赋值给data
                data["content"] = content
                
                # 首次触发切换模式 播放唤醒文案
                if check_resp["first"]:
                    # 随机获取文案 TODO: 如果此功能测试成功，所有的类似功能都将使用此函数简化代码
                    resp_json = common.get_random_str_in_list_and_format(
                        ori_list=config.get(
                            "talk", "wakeup_sleep", "wakeup_copywriting"
                        )
                    )
                    if resp_json["ret"] == 0:
                        data["content"] = resp_json["content"]
                        data["insert_index"] = -1
                        my_handle.reread_handle(data)
                else:
                    # 如果启用了“打断对话”功能
                    if config.get("talk", "interrupt_talk", "enable"):
                        # 判断文本内容是否包含中断词
                        interrupt_word = common.find_substring_in_list(
                            data["content"], config.get("talk", "interrupt_talk", "keywords")
                        )
                        if interrupt_word:
                            logger.info(f"[聊天中断] 命中中断词：{interrupt_word}")
                            # 从配置中获取需要清除的数据类型
                            clean_type = config.get("talk", "interrupt_talk", "clean_type")
                            # 各类型数据是否清除
                            message_queue = "message_queue" in clean_type
                            voice_tmp_path_queue = "voice_tmp_path_queue" in clean_type
                            stop_audio_play = "stop_audio_play" in clean_type
                            
                            clear_queue_and_stop_audio_play(message_queue, voice_tmp_path_queue, stop_audio_play)
                            return False

                    # 传递给my_handle进行进行后续一系列的处理
                    # my_handle.process_data(data, "talk")
                    response = ollama.chat(
                      'qwen2.5:7b',
                      messages=[{'role': 'user', 'content': data["content"]}]
                    )
                    data["content"] = response.message.content
                    my_handle.reread_handle(data)

                    # 单次唤醒情况下，唤醒后关闭
                    if config.get("talk", "wakeup_sleep", "mode") == "单次唤醒":
                        is_talk_awake = False
            # 睡眠情况下
            else:
                # 首次进入睡眠 播放睡眠文案
                if check_resp["first"]:
                    resp_json = common.get_random_str_in_list_and_format(
                        ori_list=config.get(
                            "talk", "wakeup_sleep", "sleep_copywriting"
                        )
                    )
                    if resp_json["ret"] == 0:
                        data["content"] = resp_json["content"]
                        data["insert_index"] = -1
                        my_handle.reread_handle(data)
    except Exception as e:
        logger.error(traceback.format_exc())

# 执行录音、识别&提交
def do_listen_and_comment(status=True):
    global \
        stop_do_listen_and_comment_thread_event, \
        faster_whisper_model, \
        sense_voice_model, \
        is_recording, \
        is_talk_awake

    try:
        is_recording = True

        config = Config(config_path)
        # 是否启用按键监听和直接对话，没启用的话就不用执行了
        # if not config.get("talk", "key_listener_enable") and not config.get("talk", "direct_run_talk"):
        #     is_recording = False
        #     return

        # 针对faster_whisper情况，模型加载一次共用，减少开销
        if "sensevoice" == config.get("talk", "type"):
            from funasr import AutoModel

            logger.info("sensevoice 模型加载中，请稍后...")
            asr_model_path = config.get("talk", "sensevoice", "asr_model_path")
            vad_model_path = config.get("talk", "sensevoice", "vad_model_path")
            if sense_voice_model is None:
                model_dir = "iic/SenseVoiceSmall"
                sense_voice_model = AutoModel(
                    model=model_dir,
                    trust_remote_code=True,
                    remote_code="./model.py",
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device="cuda:0",
                )

                logger.info("sensevoice 模型加载完毕，可以开始说话了喵~")

        while True:
            try:
                # 检查是否收到停止事件
                # if stop_do_listen_and_comment_thread_event.is_set():
                #     logger.info("停止录音~")
                #     is_recording = False
                #     break

                config = Config(config_path)

                # 根据接入的语音识别类型执行
                if config.get("talk", "type") in [
                    "baidu",
                    "faster_whisper",
                    "sensevoice",
                ]:
                    # 设置音频参数
                    FORMAT = pyaudio.paInt16
                    CHANNELS = config.get("talk", "CHANNELS")
                    RATE = config.get("talk", "RATE")

                    audio_out_path = config.get("play_audio", "out_path")

                    if not os.path.isabs(audio_out_path):
                        if not audio_out_path.startswith("./"):
                            audio_out_path = "./" + audio_out_path
                    file_name = "asr_" + common.get_bj_time(4) + ".wav"
                    WAVE_OUTPUT_FILENAME = common.get_new_audio_path(
                        audio_out_path, file_name
                    )
                    # WAVE_OUTPUT_FILENAME = './out/asr_' + common.get_bj_time(4) + '.wav'

                    frames = audio_listen(
                        config.get("talk", "volume_threshold"),
                        config.get("talk", "silence_threshold"),
                    )

                    # 将音频保存为WAV文件
                    with wave.open(WAVE_OUTPUT_FILENAME, "wb") as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b"".join(frames))

                    if config.get("talk", "type") == "sensevoice":
                        res = sense_voice_model.generate(
                            input=WAVE_OUTPUT_FILENAME,
                            cache={},
                            language=config.get("talk", "sensevoice", "language"),
                            text_norm=config.get("talk", "sensevoice", "text_norm"),
                            batch_size_s=int(
                                config.get("talk", "sensevoice", "batch_size_s")
                            ),
                            batch_size=int(
                                config.get("talk", "sensevoice", "batch_size")
                            ),
                        )

                        def remove_angle_brackets_content(input_string: str):
                            # 使用正则表达式来匹配并删除 <> 之间的内容
                            return re.sub(r"<.*?>", "", input_string)

                        content = remove_angle_brackets_content(res[0]["text"])

                        talk_handle(content)
                

                is_recording = False

                if not status:
                    return
            except Exception as e:
                logger.error(traceback.format_exc())
                is_recording = False
                return
    except Exception as e:
        logger.error(traceback.format_exc())
        is_recording = False
        return


if __name__ == "__main__":

    common = Common()
    config = Config(config_path)
    my_handle = My_handle(config_path)
    # 日志文件路径
    log_path = "./log/log-" + common.get_bj_time(1) + ".txt"

    # 按键监听相关
    do_listen_and_comment_thread = None
    stop_do_listen_and_comment_thread_event = None
    # 存储加载的模型对象
    faster_whisper_model = None
    sense_voice_model = None
    # 正在录音中 标志位
    is_recording = False
    # 聊天是否唤醒
    is_talk_awake = False

    # 待播放音频数量（在使用 音频播放器 或者 metahuman-stream等不通过AI Vtuber播放音频的对接项目时，使用此变量记录是是否还有音频没有播放完）
    wait_play_audio_num = 0
    wait_synthesis_msg_num = 0
    do_listen_and_comment()