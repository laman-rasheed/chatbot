import { createDeepAgent } from 'deepagents';

const agent = createDeepAgent();

const result = await agent.invoke({
  messages: [{ role: 'user', content: 'What is LangSmith?' }],
});