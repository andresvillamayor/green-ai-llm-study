# Listado de Prompts y Categorías — Proyecto GREEN-IA

Fuente: MT-Bench oficial (Zheng et al., NeurIPS 2023)
arXiv: 2306.05685
Subset: 5 preguntas por categoría, 8 categorías, 40 preguntas totales

---

## CODING (5 prompts)

- **ID 121:** Develop a Python program that reads all the text files under a directory and returns top-5 words with the most number of occurrences.

- **ID 122:** Write a C++ program to find the nth Fibonacci number using recursion.

- **ID 123:** Write a simple website in HTML. When a user clicks the button, it shows a random joke from a list of 4 jokes.

- **ID 124:** Here is a Python function to find the length of the longest common subsequence of two input strings. Can you identify any bug in this function?

```
def longest_common_subsequence_length(str1, str2):
    m = len(str1)
    n = len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
```

- **ID 125:** Write a function to find the highest common ancestor (not LCA) of two nodes in a binary tree.

---

## EXTRACTION (5 prompts)

- **ID 131:** Evaluate the following movie reviews on a scale of 1 to 5, with 1 being very negative, 3 being neutral, and 5 being very positive:
  1. This movie released on Nov. 18, 2019, was phenomenal. The cinematography, the acting, the plot - everything was top-notch.
  2. Never before have I been so disappointed with a movie. The plot was predictable and the characters were one-dimensional. In my opinion, this movie is the worst one to have been released in 2022.
  3. The movie was okay. There were some parts I  enjoyed, but there were also parts that felt lackluster. This is a movie that was released in Feb 2018 and seems to be quite ordinary.
Return the answer as a JSON array of integers.

- **ID 132:** Given these categories - Literature, History, Science, and Art. Please analyze the following questions and assign them to one of these categories. In your response, refrain from uttering any extraneous words. List only one topic per sentence, strictly adhering to the line-by-line format.
  1. Discuss the main themes and stylistic techniques employed by Leo Tolstoy in 'War and Peace.' How do they align with the wider social context of 19th-century Russia?
  2. Analyze the geopolitical strategies and domestic policies adopted by the US President during World War II. How did these actions shape the post-war international order?
  3. Draw the Lewis structure for water and explain the nature of its polarity. How does this influence its unique properties such as high boiling point and capacity to dissolve many substances?
  4. Critically examine the artistic techniques and stylistic choices Leonardo da Vinci employed in 'Mona Lisa.' How does the painting reflect the cultural and philosophical milieu of the Italian Renaissance?

- **ID 133:** Extract the following information from the presented texts: The name of the book, the author, the main character, the year of publication. Output in the format of "main character, book, author, year of publication", one book per line.
  a) In the realm of wizarding literature, a true standout is the work of J.K. Rowling. One of her books that left an indelible mark is 'Harry Potter and the Philosopher's Stone'. This iconic tale, published in 1997, tells the story of Harry, a young orphan who discovers his magical abilities on his 11th birthday. Soon, he finds himself at the Hogwarts School of Witchcraft and Wizardry, a place teeming with magic and adventure, located somewhere in Scotland.
  b) The magic of Middle-earth has entranced readers worldwide, thanks to the brilliance of J.R.R. Tolkien. In one of his seminal works, 'The Lord of the Rings: The Fellowship of the Ring', published in 1954, we meet Frodo Baggins, a brave hobbit tasked with the perilous quest of destroying the One Ring. The epic journey takes him from the peaceful Shire to the tumultuous regions of Middle-earth.
  c) In a galaxy far, far away, the imagination of L.E. Starlighter gives us 'The Prism Galaxy Chronicles: The Awakening of the Starcaster'. Published in 2028, the story is about Zylo, a humble spaceship mechanic, who unexpectedly discovers he's a Starcaster - a rare individual with the power to manipulate stardust. Set against the backdrop of an interstellar empire in turmoil, Zylo's destiny unfolds on numerous alien worlds, each with its unique cosmic charm.

- **ID 134:** Given the following data, identify the company with the highest profit in 2021 and provide its CEO's name:
  a) Company X, with CEO Amy Williams, reported $30 billion in revenue and a $3 billion profit in 2021.
  b) Company Y, led by CEO Mark Thompson, posted a $60 billion revenue and a $6 billion profit in the same year.
  c) Company Z, under CEO Sarah Johnson, announced a $20 billion revenue and a $7 billion profit in 2021.
  d) Company W, managed by CEO James Smith, revealed a $300 billion revenue with a $21 billion profit in 2021.
  e) Company V, with CEO Lisa Brown, reported a $200 billion revenue and a $25 billion profit in 2021.
  f) Company U, under CEO John White, posted a $180 billion revenue and a $20 billion profit in the same year.

- **ID 135:** Identify the countries, their capitals, and the languages spoken in the following sentences. Output in JSON format.
  a) Amidst the idyllic vistas, Copenhagen, Denmark's capital, captivates visitors with its thriving art scene and the enchanting Danish language spoken by its inhabitants.
  b) Within the enchanting realm of Eldoria, one discovers Avalore, a grandiose city that emanates an ethereal aura. Lumina, a melodious language, serves as the principal mode of communication within this mystical abode.
  c) Nestled amidst a harmonious blend of age-old customs and contemporary wonders, Buenos Aires, the capital of Argentina, stands as a bustling metropolis. It is a vibrant hub where the expressive Spanish language holds sway over the city's inhabitants.

---

## HUMANITIES (5 prompts)

- **ID 151:** Provide insights into the correlation between economic indicators such as GDP, inflation, and unemployment rates. Explain how fiscal and monetary policies affect those indicators.

- **ID 152:** How do the stages of life shape our understanding of time and mortality?

- **ID 153:** Discuss antitrust laws and their impact on market competition. Compare the antitrust laws in US and China along with some case studies.

- **ID 154:** Create a lesson plan that integrates drama, mime or theater techniques into a history class. Duration: 3 class periods (each lasts for 45 minutes) for 3 days
Topic: Opium Wars between China and Britain
Grade level: 9-10

- **ID 155:** Share ideas for adapting art masterpieces into interactive experiences for children. List 5 specific artworks and associated ideas.

---

## MATH (5 prompts)

- **ID 111:** The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?

- **ID 112:** A tech startup invests $8000 in software development in the first year, and then invests half of that amount in software development in the second year. What's the total amount the startup invested in software development over the two years?

- **ID 113:** In a survey conducted at a local high school, preferences for a new school color were measured: 58% of students liked the color blue, 45% preferred green, and 22% liked both colors. If we randomly pick a student from the school, what's the probability that they would like neither blue nor green?

- **ID 114:** When rolling two dice, what is the probability that you roll a total number that is at least 3?

- **ID 115:** Some people got on a bus at the terminal. At the first bus stop, half of the people got down and 4 more people got in. Then at the second bus stop, 6 people got down and 8 more got in. If there were a total of 25 people heading to the third stop, how many people got on the bus at the terminal?

---

## REASONING (5 prompts)

- **ID 101:** Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?

- **ID 102:** You can see a beautiful red house to your left and a hypnotic greenhouse to your right, an attractive heated pink place in the front. So, where is the White House?

- **ID 103:** Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?

- **ID 104:** David has three sisters. Each of them has one brother. How many brothers does David have?

- **ID 105:** Read the below passage carefully and answer the questions with an explanation:
  At a small company, parking spaces are reserved for the top executives: CEO, president, vice president, secretary, and treasurer with the spaces lined up in that order. The parking lot guard can tell at a glance if the cars are parked correctly by looking at the color of the cars. The cars are yellow, green, purple, red, and blue, and the executives' names are Alice, Bert, Cheryl, David, and Enid.
  * The car in the first space is red.
  * A blue car is parked between the red car and the green car.
  * The car in the last space is purple.
  * The secretary drives a yellow car.
  * Alice's car is parked next to David's.
  * Enid drives a green car.
  * Bert's car is parked between Cheryl's and Enid's.
  * David's car is parked in the last space.
Question: What is the name of the secretary?

---

## ROLEPLAY (5 prompts)

- **ID 91:** Pretend yourself to be Elon Musk in all the following conversations. Speak like Elon Musk as much as possible. Why do we need to go to Mars?

- **ID 92:** Embrace the role of Sheldon from "The Big Bang Theory" as we delve into our conversation. Don't start with phrases like "As Sheldon". Let's kick things off with the following question: "What is your opinion on hand dryers?"

- **ID 93:** Imagine yourself as a doctor tasked with devising innovative remedies for various ailments and maladies. Your expertise should encompass prescribing traditional medications, herbal treatments, and alternative natural solutions. Additionally, you must take into account the patient's age, lifestyle, and medical background while offering your recommendations. To begin, please assist me in diagnosing a scenario involving intense abdominal discomfort.

- **ID 94:** Please take on the role of a relationship coach. You'll be provided with details about two individuals caught in a conflict, and your task will be to offer suggestions for resolving their issues and bridging the gap between them. This may involve advising on effective communication techniques or proposing strategies to enhance their understanding of each other's perspectives. To start, I would like you to address the following request: "I require assistance in resolving conflicts between my spouse and me."

- **ID 95:** Please assume the role of an English translator, tasked with correcting and enhancing spelling and language. Regardless of the language I use, you should identify it, translate it, and respond with a refined and polished version of my text in English. Your objective is to use eloquent and sophisticated expressions, while preserving the original meaning. Focus solely on providing corrections and improvements. My first request is "衣带渐宽终不悔 为伊消得人憔悴".

---

## STEM (5 prompts)

- **ID 141:** In the field of quantum physics, what is superposition, and how does it relate to the phenomenon of quantum entanglement?

- **ID 142:** Consider a satellite that is in a circular orbit around the Earth. The speed of the satellite decreases. What will happen to the satellite's orbital radius and period of revolution? Please justify your answer using principles of physics.

- **ID 143:** Photosynthesis is a vital process for life on Earth. Could you outline the two main stages of photosynthesis, including where they take place within the chloroplast, and the primary inputs and outputs for each stage?

- **ID 144:** What is the central dogma of molecular biology? What processes are involved? Who named this?

- **ID 145:** Describe the process and write out the balanced chemical equation for the reaction that occurs when solid calcium carbonate reacts with hydrochloric acid to form aqueous calcium chloride, carbon dioxide, and water. What type of reaction is this, and what observations might indicate that the reaction is taking place?

---

## WRITING (5 prompts)

- **ID 81:** Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.

- **ID 82:** Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.

- **ID 83:** Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings to effectively compare and contrast the features, performance, and user experience of the two models. Please answer in fewer than 200 words.

- **ID 84:** Write a persuasive email to convince your introverted friend, who dislikes public speaking, to volunteer as a guest speaker at a local event. Use compelling arguments and address potential objections. Please be concise.

- **ID 85:** Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer than two paragraphs.
