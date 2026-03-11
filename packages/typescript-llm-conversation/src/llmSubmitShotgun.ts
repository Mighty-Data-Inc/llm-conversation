import { AIClientLike, identifyLLMProvider } from './llmProviders.js';
import { llmSubmit, LLMSubmitOptions } from './llmSubmit.js';

/**
 * Implements the "shotgun" strategy for {@link llmSubmit}: sends `numBarrels`
 * parallel requests with identical inputs, then asks the model to examine all
 * responses and reconcile them into a single authoritative answer.
 *
 * This improves output quality for tasks where the model benefits from
 * exploring multiple reasoning paths simultaneously.
 *
 * @param messages - The conversation history to send to each worker.
 * @param aiClient - The LLM provider's client instance.
 * @param options - Submit options (the `shotgun` field is stripped before
 *   forwarding to avoid infinite recursion).
 * @param numBarrels - Number of parallel worker requests to fire.
 * @returns The reconciled response from the model.
 */
export const llmSubmitShotgun = async (
  messages: unknown[],
  aiClient: AIClientLike,
  options: LLMSubmitOptions,
  numBarrels: number
): Promise<
  string | Record<string, unknown> | unknown[] | number | boolean | null
> => {
  messages = JSON.parse(JSON.stringify(messages));
  options = JSON.parse(JSON.stringify(options));

  // Delete the shotgun option from the options passed to
  // llmSubmitShotgun to avoid infinite recursion.
  delete options.shotgun;

  if (numBarrels <= 1) {
    // No need for shotgun logic if only 1 barrel requested.
    return llmSubmit(messages, aiClient, options);
  }

  // Remove the shotgun option before passing to llmSubmit to avoid
  // infinite recursion!
  delete options.shotgun;

  const convoShotgun: unknown[][] = [];
  for (let i = 0; i < numBarrels; i += 1) {
    const convoBarrel = JSON.parse(JSON.stringify(messages));
    convoShotgun.push(convoBarrel);
  }
  const resultsRaw: unknown[] = await Promise.all(
    convoShotgun.map((convoBarrel) => llmSubmit(convoBarrel, aiClient, options))
  );
  const resultStrings = resultsRaw.map((result) => JSON.stringify(result));

  // Build the reconciliation conversation on top of a fresh copy of the original messages.
  const reconcileMessages: unknown[] = JSON.parse(JSON.stringify(messages));

  reconcileMessages.push({
    role: 'system',
    content: `
SYSTEM MESSAGE:
In order to produce better results, we submitted this request/question/command/etc.
to ${numBarrels} worker threads in parallel.
The system will now present each of their responses, wrapped in JSON.
The user or developer will not see these responses -- they are only for you, the assistant, 
to examine and reconcile. Think of them as brainstorming or scratchpads.
`,
  });
  resultStrings.forEach((resultString, index) => {
    reconcileMessages.push({
      role: 'system',
      content: `WORKER ${index + 1} RESPONSE:\n\n\n${resultString}`,
    });
  });

  reconcileMessages.push({
    role: 'system',
    content: `
Focus on the differences and discrepancies between the workers' responses. Where do they agree?
Where do they disagree? In the areas where they disagree, which worker's argument is most
consistent with the data you've been shown?

Remember, this is an adjudication, not a democracy -- you should carefully examine the data
presented in the conversation and use your best judgment to determine which worker is most
likely to be correct, even if they're in the minority. Evaluate their answers carefully against
the source data. If multiple workers produced different answers, then clearly there is something
subtle, deceptive, or misleading about the question or the data, and you should be especially
careful to scrutinize the workers' reasoning and the evidence they present for their answers.
At least one of them must be wrong; don't fall into the same trap he did.
`,
  });

  // This is a chain-of-thought ponderance. We specifically do not want a JSON
  // response here, because we want the model to be able to freely reason and
  // draw conclusions without being constrained by JSON syntax. The final answer
  // will be produced in the next step, where we will ask the model to produce
  // the same format as it was originally given (text or JSON).
  const sPonderReply = await llmSubmit(reconcileMessages, aiClient, {
    ...options,
    jsonResponse: undefined,
  });
  reconcileMessages.push({ role: 'assistant', content: sPonderReply });

  reconcileMessages.push({
    role: 'system',
    content: `
Having seen and reconciled the workers' responses, you are now ready to craft a proper reply to
the question/request/command/etc. This response that you craft now is the one that will be
presented to the user or developer -- it should not directly reference the workers' responses,
but should instead be a fully self-contained and complete answer that draws on the insights
you've gained from examining the workers' responses.
`,
  });

  return llmSubmit(reconcileMessages, aiClient, options);
};
