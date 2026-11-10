# The Relay --- Assignment Context

## Source of truth

This document records the requirements and submission guidelines in
**The Relay --- Assignment brief** (4 pages). It is intended as working
context for future work on this assignment.

**Important:** Treat the PDF as the source of truth. Do not silently
change its requirements, invent missing requirements, or present
implementation suggestions as if they were specified by the brief.

------------------------------------------------------------------------

## 1. Problem scenario

A company runs an assistant that answers engineers' questions about a
small library of research papers. The assistant chains several model
calls and tools. It:

1.  Works out what the person is asking.
2.  Retrieves passages from the papers.
3.  Decides what to do.
4.  Drafts a cited reply.
5.  Checks the reply before sending it.

The system feels fast on a quiet afternoon, but the assignment asks what
happens at launch.

The brief describes these operating conditions:

-   Requests arrive in bursts, usually after something goes wrong for
    many users at once.
-   Most requests need a few steps; some need many; a few loop.
-   The model provider limits requests per minute, tokens per minute and
    requests per day.
-   Limits differ by plan and model, and can change without notice.
-   Tools time out.
-   Some calls are slow for no visible reason.
-   Some calls succeed slowly instead of failing quickly.
-   People are waiting and give up sooner than expected.

The company wants every reply fast, every reply good, the bill small,
and nobody turned away. The brief explicitly says it cannot have all
four. It also says the system must never send a wrong or embarrassing
answer just to stay fast.

The details are intentionally incomplete. The candidate must decide what
matters, choose their own numbers, make real engineering decisions, and
explain where one goal is sacrificed for another.

## 2. What to build

Build a small working version of the pipeline in **Python, C++ or Go**
(candidate's choice).

Mandatory pipeline requirements:

-   At least four steps.
-   At least one tool call.
-   At least one branch or loop.
-   One step must retrieve passages from the library below.

### The library: exact papers and versions

The assistant must answer only from these three arXiv papers. Fetch the
exact versions specified:

1.  **Attention Is All You Need** --- arXiv `1706.03762`, version 7.
2.  **Moshi: a speech-text foundation model for real-time dialogue** ---
    arXiv `2410.00037`, version 1.
3.  **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling**
    --- arXiv `2507.07955`, version 2.

Every answer must cite where in the papers it came from. If the papers
do not say, the assistant must say so instead of guessing. The brief
says reviewers will ask questions that cannot be answered from memory
and questions the papers do not answer at all.

### Real hosted model

-   Use a real hosted model on **Groq's free tier** for the model calls.
-   Use the candidate's own free account and key.
-   Find out the limits from Groq's documentation and from what Groq
    sends back when a limit is reached.

### Stand-in model service

Because a free tier will not survive heavy load, also write a stand-in
model service that the candidate controls.

It must be able to:

-   Be slow.
-   Fail.
-   Refuse because a limit was reached.
-   Become slower under load.
-   Send its reply a piece at a time (streaming).

Calibrate the stand-in from measurements taken on the real service, and
state where it is least like the real thing.

## 3. What to run

First, send a modest number of real requests to learn how the real
service behaves:

-   How its latency is distributed.
-   What its limits are.
-   What it does when a limit is reached.

Then run the pipeline under **at least three load shapes** of the
candidate's own design. Examples in the brief:

-   Steady.
-   A sudden burst.
-   Sustained overload.

Use the stand-in for loads the real service cannot take.

Keep the raw results for both real-service and stand-in runs.

------------------------------------------------------------------------

## 4. Written report requirements

The brief says the write-up has a **maximum of two pages of writing**,
plus code, raw results and conversation history.

It requires the following headings, in the stated order, and says a page
that skips one is incomplete.

### Required numbered headings

1.  **What I built and why.** The design, and the one decision that
    shaped it most.
2.  **The real service and my stand-in.** The limits found in the real
    service and how they were found. How the stand-in behaves, how it
    was calibrated from real measurements, and where it differs most
    from the real thing.
3.  **Results.** Time to first token and total time, each at the median
    and at the slow end (state which percentiles and how many requests),
    failures, and given-up requests, for each load shape. State which
    runs were real and which used the stand-in. Put every number in the
    results tables. Include a number reviewers could not have guessed
    without running it. Also report answer quality on at least 15
    questions written by the candidate, and tokens used per answer.
4.  **Where the time goes.** For one slow request, split its latency
    into waiting, working and retrying, and say whether the delay came
    before or after the first token. Show how this is known.
5.  **When a limit is hit.** What the system does at the per-minute
    limit and at the per-day limit. Who is delayed, who is dropped, who
    gets a weaker answer, and why.
6.  **Retries.** The retry policy, and one case where retrying made
    things worse.
7.  **What the person sees.** At 3pm the daily budget is gone. Describe
    the reply a waiting person receives.
8.  **What I'm unsure about, what I left out, what I learned.** The
    least certain assumption and what breaks if it is wrong. Two things
    deliberately not built. One thing that was new to the candidate,
    taught in their own words.

### Two additional headings to insert after heading 5

-   **Retrieval: quality versus cost.** Try at least three settings for
    how the papers are split and how much is fetched (piece size, number
    of pieces, and any re-ranking). For each, report answer quality,
    tokens per answer, and time to first token. Choose one setting and
    say why, and say what happens to retrieval when the limit is close.
-   **How this library will mislead a naive retriever.** Name three
    ways, each with an example question where it goes wrong.

### One additional heading

-   **Your feedback on Vaani.** Visit `https://www.vaani.dev/` and give
    honest feedback in a few lines: what stood out, what was confusing,
    and one thing to change. The brief says there is no right answer.

### Final section

Finish with a section called **How I worked with the assistant**.

It must include:

-   The real conversation history with the assistant, not a cleaned-up
    version; or the prompts if the tool cannot export the history.
-   In half a page:
    -   The three prompts that mattered most and why.
    -   One thing the assistant got wrong that the candidate caught, and
        how it was caught.
    -   One decision where the candidate overruled the assistant.
    -   What in the submission is distinctly the candidate's own.

The brief says there is no single correct answer. It says reviewers care
whether choices follow from measurements and whether the candidate can
say what those choices cost.

**Ordering note:** The PDF first lists numbered headings 1--8 in order,
then instructs the candidate to insert the two retrieval/library
headings after heading 5, then adds the Vaani heading, and says to
finish with "How I worked with the assistant." Preserve those placement
instructions when assembling the report.

------------------------------------------------------------------------

## 5. Required results tables

The brief says:

-   Every number goes in these tables.
-   The tables must be completely filled in.
-   Write **"not measured"** where something was not measured.
-   Reviewers read these tables first and recompute them from the raw
    file, so the tables must match the raw data.
-   Use milliseconds for time.
-   Use one decimal place for percentages.
-   Copy the tables into the write-up.

### Table 1: Limits found in the real service

Required columns:

  -----------------------------------------------------------------------
  Limit             Value you found   How you found it  Confidence (high,
                                                        medium, low)
  ----------------- ----------------- ----------------- -----------------
  Requests per                                          
  minute                                                

  Tokens per minute                                     

  Requests per day                                      

  Tokens per day                                        

  Requests in                                           
  flight at once                                        

  What it returns                                       
  when a limit is                                       
  hit                                                   

  Time until it                                         
  recovers                                              
  -----------------------------------------------------------------------

### Table 2a: Latency --- one row per load scenario

The brief requires at least these scenarios:

-   Steady.
-   Burst.
-   Sustained overload.
-   Per-minute limit halved.
-   Daily budget exhausted.

Required columns:

  ---------------------------------------------------------------------------
  Scenario   Real or       Requests    Duration     Offered First    Total
             stand-in          sent         (s)   rate (per token    time p50
                                                       min) p50 /    / p95 /
                                                            p95 /    p99 (ms)
                                                            p99 (ms) 
  ---------- ---------- ----------- ----------- ----------- -------- --------

  ---------------------------------------------------------------------------

### Table 2b: Outcomes --- same rows as Table 2a

Required columns:

  ---------------------------------------------------------------------------------------
  Scenario   Succeeded % Rate-limited % Timed out   Retries      Peak    Tokens  Cost per
                                         or given       per     queue       per     1,000
                                             up %   request    length    answer   answers
  ---------- ----------- -------------- --------- --------- --------- --------- ---------

  ---------------------------------------------------------------------------------------

### Table 3: Retrieval settings --- at least three

Required columns:

  --------------------------------------------------------------------------------
  Setting      Questions   Correct   Wrong %   Correctly    Tokens First   Total
  (piece           asked and cited             refused %       per token   time
  size,                          %                          answer p50 /   p50 /
  pieces                                                           p95     p95
  fetched,                                                         (ms)    (ms)
  re-rank                                                                  
  yes or no)                                                               
  ---------- ----------- --------- --------- ----------- --------- ------- -------

  --------------------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Raw results file

The required raw file must be named exactly:

`results.csv`

It must contain **one row per request** and these columns:

1.  `request_id`
2.  `scenario`
3.  `source` (real or stand-in)
4.  `start_time`
5.  `first_token_ms`
6.  `total_ms`
7.  `status` (`ok`, `rate_limited`, `timeout`, `error`)
8.  `retries`
9.  `tokens_in`
10. `tokens_out`
11. `retrieval_setting`
12. `answer_grade` (`correct`, `wrong`, `refused_correctly`,
    `refused_wrongly`)

Every figure in the report tables must be computable from this file.

------------------------------------------------------------------------

## 7. Ground rules and submission constraints

-   Submit within **4 days of receiving the brief**.
-   Expect about **6 to 8 hours of real work**, spread however the
    candidate likes.
-   Maximum **2 pages of writing**, plus code, raw results and
    conversation history.
-   Use the candidate's own free account.
-   Do not include the Groq key in the submission.
-   The account's daily limit is shared by everything run, so plan
    requests; running out is part of the problem.
-   Stay within the provider's terms.
-   Do not use extra accounts or any other trick to get around provider
    limits. Working inside the limits is the point.
-   AI coding assistants are welcome and expected by the brief.
-   The reviewers are evaluating how well the candidate directs the
    assistant: framing the problem, checking its output, and overruling
    it where appropriate.
-   The reviewers are also looking for the candidate's own contribution:
    choices, measurements and reasoning that someone else's prompt would
    not have produced.
-   A short live conversation follows, where the candidate explains the
    results and changes the system. Submit only what the candidate can
    defend.
-   The brief says the candidate does not need to cover every technique;
    pick the few that matter for the measurements and go deep.

------------------------------------------------------------------------

## 8. Key constraints to preserve during future work

This is a compact checklist of requirements already stated above; it
does not add new requirements.

-   [ ] Implement the pipeline in Python, C++ or Go.
-   [ ] Include at least four steps, one tool call, and one branch or
    loop.
-   [ ] Retrieve passages from the three exact paper versions.
-   [ ] Cite every answer and correctly say when the papers do not
    provide the answer.
-   [ ] Use a real Groq free-tier hosted model with the candidate's own
    account.
-   [ ] Measure the real service modestly: latency distribution, limits,
    and limit-hit behaviour.
-   [ ] Build the controllable stand-in with delay, failure, limit
    refusal, load-dependent slowdown and piece-by-piece replies.
-   [ ] Calibrate the stand-in from real measurements and explain its
    biggest mismatch with the real service.
-   [ ] Run at least three self-designed load shapes; include the five
    scenarios required by Table 2a.
-   [ ] Preserve raw results for real and stand-in runs.
-   [ ] Evaluate answer quality on at least 15 self-written questions.
-   [ ] Compare at least three retrieval settings.
-   [ ] Complete all three report tables; use "not measured" where
    appropriate.
-   [ ] Ensure every reported table number can be recomputed from
    `results.csv`.
-   [ ] Include the required report headings and Vaani feedback.
-   [ ] Include the actual assistant conversation history or prompts,
    plus the half-page reflection.
-   [ ] Keep the written report to a maximum of two pages.
-   [ ] Submit within four days of receiving the brief.
-   [ ] Do not include the API key or bypass provider limits.
-   [ ] Be able to explain and defend the work in the live conversation.
