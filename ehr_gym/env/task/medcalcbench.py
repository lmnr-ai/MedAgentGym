import os
import re
from datetime import date, datetime
from .base import AbstractEHRTask
import json

# 55 of the 57 calculators produce a number, graded against [Lower Limit, Upper
# Limit]. The other two produce a calendar date ("Estimated Due Date",
# "Estimated of Conception") or a gestational age ("Estimated Gestational Age"),
# and for those the limits are the answer repeated. Comparing them as floats is
# what made all 60 of those datapoints unpassable, so each answer shape gets its
# own parser and comparison.
DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%Y-%m-%d")
DATE_RE = re.compile(r"\d{1,4}[/-]\d{1,2}[/-]\d{2,4}")
GESTATIONAL_RE = re.compile(r"(\d+)\s*weeks?\D+?(\d+)\s*days?", re.I)


def parse_date(value: str) -> date | None:
    match = DATE_RE.search(value)
    if match is None:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(match.group(0), fmt).date()
        except ValueError:
            continue
    return None


def parse_gestational_age(value: str) -> tuple[int, int] | None:
    match = GESTATIONAL_RE.search(value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_number(value: str) -> float | None:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None

overall_information = """
You work in a hospital, and a common task in your work is to calculate some biological values of your patients. To do this, you need to identify from clinical notes what information is relevant, before using your clinical knowledge to calculate.
Instructions to calculate is listed below {}.
"""

instruction = """You work in a hospital, and a common task in your work is to calculate some biological values of your patients. 
To do this, you need to identify from clinical notes what information is relevant, before using your clinical knowledge to calculate.
And then write a Python code to calculate the value.
In the code, please use the variable 'answer' to store the answer of the code.
In the main function, please print the final answer of the code without any other text.
"""
# Hints to calculate is listed below: 
# {overall}.
# """

class MedCalBenchTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the MedCalcBench EHR data.

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
        calculator_instruction_path: str = None,
        debugger_config: dict = None,
        mode: str = "test",
    ) -> None:
        super().__init__(task_id=task_id)
        self.task_id = task_id
        self.task_list = None
        self.data_path = data_path
        self.calculator_instruction_path = calculator_instruction_path
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
        return f"EHRGym.medcalcbench.{formatted_name}"
    
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
            if self.mode == 'test':
                # Try multiple possible file names
                possible_files = ['test_tasks.jsonl', 'test.jsonl']
            else:
                possible_files = ['train_tasks_all.jsonl', 'train.jsonl']
            
            task_path = None
            for task_file in possible_files:
                candidate_path = os.path.join(self.data_path, task_file)
                if os.path.exists(candidate_path):
                    task_path = candidate_path
                    break
            
            if task_path is None:
                raise FileNotFoundError(f"No task file found in {self.data_path}. Tried: {possible_files}")
            
            self.task_list = []
            with open(task_path, 'r') as f:
                for line in f:
                    self.task_list.append(json.loads(line))
        task_data = self.task_list[self.task_id]
        self.context = task_data['Patient Note']
        self.question = task_data['Question']
        self.answer = task_data['Ground Truth Answer']
        self.lower_limit = task_data.get('Lower Limit')
        self.upper_limit = task_data.get('Upper Limit')
        self.calculator = task_data['Calculator Name']
        calculator_path = os.path.join(self.data_path, "calculation_method.jsonl")
        self.calculator_info = {}
        with open(calculator_path, 'r') as f:
            for line in f:
                calc_data = json.loads(line)
                if not calc_data['Calculator'] in self.calculator_info:
                    self.calculator_info[calc_data['Calculator']] = calc_data["Short Summary"]
        self.calculator_instruction = self.calculator_info[self.calculator]

        # configure the task
        goal, info = self.setup_goal()
        return goal, info

    
    def setup_goal(self) -> tuple[str, dict]:
        """
        Set up the goal for the task
        """
        super().setup_goal()
        # get the task configuration - include patient note in the goal
        self.goal = f"""Write a python code to solve the given question. Use the variable 'answer' to store the answer of the code.

Patient Note:
{self.context}

Question: {self.question}
"""
        info = {}
        return self.goal, info

    def _get_obs(self) -> dict:
        obs = {}
        obs["type"] = "initial_observation"
        obs["info"] = {}
        obs["info"]["overall"] = overall_information.format(self.calculator_instruction)
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = instruction # .format(overall=self.calculator_instruction)
        return obs
    

    def compare(self, pred: str) -> bool:
        """Compare the agent's printed answer to the ground truth.

        The comparison follows the shape of the ground truth, not of the
        prediction, so that a date question is never accidentally graded as a
        number. Raises ValueError when the prediction cannot be read at all.
        """
        answer = self.answer[0] if isinstance(self.answer, list) else self.answer
        pred = (pred or "").strip()

        expected_date = parse_date(answer)
        if expected_date is not None:
            actual = parse_date(pred)
            if actual is None:
                raise ValueError(f"expected a date in M/D/Y format, got {pred!r}")
            return actual == expected_date

        expected_age = parse_gestational_age(answer)
        if expected_age is not None:
            actual = parse_gestational_age(pred)
            if actual is None:
                raise ValueError(f"expected a '<N> weeks, <M> days' answer, got {pred!r}")
            return actual == expected_age

        value = parse_number(pred)
        if value is None:
            raise ValueError(f"expected a number, got {pred!r}")
        lower, upper = parse_number(str(self.lower_limit)), parse_number(str(self.upper_limit))
        if lower is not None and upper is not None:
            return lower <= value <= upper
        expected = parse_number(answer)
        if expected is None:
            raise ValueError(f"ground truth {answer!r} is not a number")
        return abs(value - expected) <= abs(expected) * 0.05

    def validate(self, chat_messages, obs):
        """
        Validate the task

        Parameters:
        -----------------
        chat_messages: list
            List of chat messages
        obs: dict
            Observation dictionary
        """
        
        if obs["type"] == "code_execution":
            pred = obs.get("stdout", obs["env_message"])
            try:
                correctness = self.compare(pred)
            except ValueError as e:
                return (
                    0,
                    False,
                    "The code encountered with errors",
                    {"message": f"The code encountered with errors during evaluation. There seems to be something wrong with the final answer or not print it. Can you check the error message and try to fix it?\nError Message: {str(e)}"}
                )

            if correctness:
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
    
