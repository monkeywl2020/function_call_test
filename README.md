本工是一个LLM大语言模型适配的 Function call测试工程，可以对接多种模型，采用的是非流式的输出。可以用来测试大模型的 Function call功能是否正常。主要代码 function_call_agent.py 是入口代码。config_list 是模型配置。

1： generate_llm_reply 是调用大模型适配文件目录下的方法与大模型进行交互。 大模型适配文件  function_call\models 在这个目录下。生成的响应如果是tool call会转换成标准的 openAI响应格式。

2：generate_tool_calls_reply 是解析大模型响应调用Function 并且将调用结果转成 标准openAI格式。

while循环是根据响应结果看是否需要调用 generate_tool_calls_reply，直到最终返回的内容不带 tool call内容就会结束。
