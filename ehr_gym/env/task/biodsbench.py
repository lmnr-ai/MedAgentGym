import ast
import os
from .base import AbstractEHRTask
from .substitution import DEFAULT_PATTERN
import json

instruction = """You are an biomedical expert in writing bioinformatics code and answer questions accordingly.
Your objective is to write a python code to solve the given question.
Please only write the code, do not include any other text.
All the required data are stored in the directory: {dataset_path}
Your code is appended to the setup code shown below and then checked with
assertions, so define the variables the question asks for at module level.
"""

dataframe_information = """The data files available to you are described below:
{dataframes}
"""

code_history_information = """Your code will be appended to the following setup code, which has already run:
<code>
{code_history}
</code>
"""

# The tasks were written against a container that mounted one study at /workdir.
# We keep every study side by side instead, so the paths in the questions, in the
# setup code and in the assertions are rewritten to the study's own directory.
WORKDIR_PREFIX = "/workdir"


def _as_list(value) -> list[str]:
    """Several fields are lists that were serialized with repr() into a string."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value] if value else []
    return parsed if isinstance(parsed, list) else [str(parsed)]

class BioDSBenchTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the Biocoder data.

    Class sttributed:
    -----------------
    config_path: str
        Path to the configuration file
    
    Parameters:
    -----------------
    task_id: int
        The id of the task inside the data.
    
    """
    permitted_actions = ['validate_code', 'debug', 'terminal']
    def __init__(
        self,
        task_id: int,
        data_path: str = None,
        debugger_config: dict = None,
        mode: str = "test",
    ) -> None:
        super().__init__(task_id=task_id)
        self.task_id = task_id
        self.task_list = None
        self.data_path = data_path
        self.debugger_config = debugger_config
        self.mode = mode
    
    @classmethod
    def get_task_id(cls):
        # Get the class name and remove the word 'Task' from the end if it exists
        class_name = cls.__name__.replace("Task", "")
        # Convert CamelCase to hyphen-separated format
        formatted_name = "".join(
            ["-" + c.lower() if c.isupper() else c for c in class_name]
        ).lstrip("-")
        return f"EHRGym.biocoder.{formatted_name}"
    
    def setup(self) -> tuple[str, dict]:
        """
        Set up the task

        Parameters:
        -----------------
        data_path: str
            Path to the data directory
        """
        # locate the task
        if self.task_list is None:
            if self.mode == "test":
                task_file = "test_tasks.jsonl"
            else:
                task_file = 'train_tasks.jsonl'
            task_path = os.path.join(self.data_path, task_file)
            self.task_list = []
            with open(task_path, 'r') as f:
                for line in f:
                    self.task_list.append(json.loads(line))
        task_data = self.task_list[self.task_id]
        self.study_id = str(task_data['study_ids'])
        # Absolute: the questions, the setup code and the assertions all carry this
        # path into code that runs in the rollout's own scratch directory, not in
        # the repo root.
        self.dataset_path = os.path.abspath(os.path.join(self.data_path, 'data', self.study_id))
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(
                f"No data for study {self.study_id} at {self.dataset_path}. "
                f"Run scripts/fetch_biodsbench_data.py first."
            )

        localize = lambda text: text.replace(WORKDIR_PREFIX, self.dataset_path)
        self.question = localize(task_data['queries'] + '\n' + task_data['cot_instructions'])
        self.dataframes = localize(task_data['dataframes'])
        self.code_history = localize("\n".join(_as_list(task_data['code_histories'])))

        # The agent's submission is spliced between the setup code and the
        # assertions, so the assertions -- not merely "the code ran" -- decide the
        # reward, and the agent only has to write the step the question asks for.
        self.context = "\n".join(
            [self.code_history, DEFAULT_PATTERN, localize(task_data['test_cases'])]
        )
        self.context_pattern = DEFAULT_PATTERN
        self.code_id = '{}-{}'.format(task_data['study_ids'], task_data['question_ids'])
        goal, info = self.setup_goal()

        return goal, info

    def setup_goal(self) -> tuple[str, dict]:
        """
        Set up the goal and info for the task

        Parameters:
        -----------------
        data_path: str
            Path to the data directory
        """
        # The assertions reference variables that the setup code defines, so the
        # agent has to see it; upstream built these strings and then dropped them.
        self.goal = "\n".join(
            [
                self.question,
                dataframe_information.format(dataframes=self.dataframes),
                code_history_information.format(code_history=self.code_history),
            ]
        )
        info = {
            "code_id": self.code_id,
            "study_id": self.study_id,
            "dataset_path": self.dataset_path,
        }
        return self.goal, info

    def _get_obs(self) -> dict:
        obs = {}
        obs["type"] = "initial_observation"
        obs["info"] = {}
        obs["info"]["question"] = self.question
        obs["info"]["code_id"] = self.code_id
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = instruction.format(dataset_path=self.dataset_path)

        return obs

    def validate(self, chat_messages, obs):
        if obs["type"] == "code_execution":
            if obs["status"] == "SUCCESS":
                return (
                    1,
                    True,
                    "The answer is correct",
                    {"message": "The question is correctly solved."}
                )
            else:
                return (
                    0,
                    False,
                    "The answer is incorrect",
                    {"message": "The question is not correctly solved. Can you think about whether there might be some mistakes in the previous code?"}
                )
        elif obs["type"] == "error_message":
            return (
                0,
                False,
                "The code encountered with errors",
                {"message": f"The code encountered with errors. Can you check the error message and try to fix it?\nError Message: {obs['message']}"}
            )
        else:
            return (
                0,
                False,
                "",
                {"message": obs['env_message']}
            )
