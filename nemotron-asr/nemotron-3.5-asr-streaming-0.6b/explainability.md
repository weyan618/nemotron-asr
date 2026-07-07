Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Task/Domain:                                                                   |  Speech Transcription
Model Type:                                                                                            |  FastConformer-CacheAware-RNNT
Intended Users:                                                                                        |  This model is intended for developers and data scientists building interactive call centers, virtual assistants, and language learning assistants.
Output:                                                                                                |  Transcribed text with timestamps and confidence scores
Describe how the model works:                                                                          |  Model transcribes audio input into text for the input language
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  |  Age, Gender, National Origin
Technical Limitations & Mitigation:                                                                    |  Transcripts may not be 100% accurate. Accuracy varies depending on the characteristics of the input audio, such as domain, use case, accent, noise, speech type, and speech context. 
Verified to have met prescribed NVIDIA quality standards:  |  Yes
Performance Metrics:                                                                                   |  Word Error Rate (WER), Silence Robustness (Characters/mins of silent audio), Latency (in milliseconds), Throughput (Total audio processed per unit of time)
Potential Known Risks:                                                                                 |  Not recommended for word-for-word transcription as accuracy varies based on the characteristics of input audio (domain, use case, accent, noise, speech type, and context of speech)
Licensing:                                                                                             | Governing Terms: Use of the model is governed by the [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license.

 
