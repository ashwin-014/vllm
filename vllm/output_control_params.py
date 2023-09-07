"""Primitives for output control of LLMs."""
import json
import torch
from typing import List, Dict


class OutputControlConfig:
    """The configuration of output control.

    Args:
        schema: The schema of the output control.
        logits_warper: The logits warper to use.
        options: The options of the output control.
    """

    def __init__(self) -> None:
        schema: Dict
        # selection: SelectionLogitsWarper
        options: List[str]

    @classmethod
    def from_output_control_args(cls) -> "OutputControlConfig":
        obj = cls()
        return obj


class OutputControlParams:
    def __init__(self, output_guidance_config, tokenizer) -> None:
        self.output_guidance_config = output_guidance_config
        self.tokenizer = tokenizer
        self.type = output_guidance_config["type"]
        self.schema = output_guidance_config["schema"]

        # get the eos_token
        # not handling nested json at the moment
        self.end_token = self.tokenizer.eos_token
        self.current_step = 0
        self.current_substep = 0
        self.current_step_done = False
        self.all_steps_done = False
        self.control_state = []
        if self.type == "json":
            for j, (k, v) in enumerate(self.schema.items()):
                prefix, suffix = "", ""
                if j == 0:
                    prefix = "{"

                tokens = self.tokenizer(f"{prefix}\"{k}\": \"{suffix}")["input_ids"]  # [1:]
                self.control_state.append({
                    "target_term": tokens,
                    "selected_target": 0,
                    "tokenized_subterm": None,
                    "sub_step_number": 0,
                    "type": "key",
                    "end_token": tokens[-1]
                })
                # if select in v, add i
                # 
                # n select control state. Else add normal value control state
                if "select: " in v:
                    # print("select v: ", v)
                    v = json.loads(v.replace("select: ", ""))
                    print("select v: ", type(v), v)
                    self.control_state.append({
                        "target_term": [self.tokenizer(sv)["input_ids"][1:] + self.tokenizer("\",")["input_ids"][1:] for sv in v],
                        "selected_target": 0,
                        "tokenized_subterm": None,
                        "sub_step_number": 0,
                        "type": "select",
                        "end_token": self.tokenizer.eos_token  # eos is checked during selection
                    })
                else:
                    print("normal v: ", v)
                    self.control_state.append({
                        "target_term": "",
                        "tokenized_subterm": None,
                        "sub_step_number": 0,
                        "type": "value",
                        "end_token": self.tokenizer(",")["input_ids"] + self.tokenizer("\",")["input_ids"] \
                            + self.tokenizer("\"\n")["input_ids"] + self.tokenizer("\"\n")["input_ids"] \
                                + [29908] + [613]
                    })
            self.control_state.append({
                "target_term": self.tokenizer("}")["input_ids"][1:],
                "selected_target": 0,
                "tokenized_subterm": None,
                "sub_step_number": 0,
                "type": "key",
                "end_token": tokens[-1]
            })

        elif self.type == "select":
            # tokens = self.tokenizer("some text")
            # print(tokens, dir(tokens))
            # {'input_ids': [1, 777, 1426], 'attention_mask': [1, 1, 1]}
            # ['_MutableMapping__marker', '__abstractmethods__', '__class__', '__contains__', '__copy__', '__delattr__',
            # '__delitem__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__',
            # '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__',
            # '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__setattr__',
            # '__setitem__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl',
            # '_encodings', '_n_sequences', 'char_to_token', 'char_to_word', 'clear', 'convert_to_tensors', 'copy', 'data', 'encodings',
            # 'fromkeys', 'get', 'is_fast', 'items', 'keys', 'n_sequences', 'pop', 'popitem', 'sequence_ids', 'setdefault', 'to',
            # 'token_to_chars', 'token_to_sequence', 'token_to_word', 'tokens', 'update', 'values', 'word_ids', 'word_to_chars',
            # 'word_to_tokens', 'words']
            self.control_state = [{
                "target_term": [self.tokenizer(v)["input_ids"][1:] for v in self.schema],
                "selected_target": 0,
                "tokenized_subterm": None,
                "sub_step_number": 0,
                "type": "select",
                "end_token": self.tokenizer.eos_token  # eos is checked during selection
            }]

        print("finished init of output control params")

    def get_schema(self):
        return self.schema

    def get_control_type(self):
        return self.type

    def get_state(self, key_step):
        return self.control_state[key_step]["sub_step_number"]  # [token_number]

    def set_next_step(self, key_step, tokenized_subterm, max_len_reached=False):
        if self.control_state[self.current_step]["type"] == "select":
            print(f"incrementing select {self.current_substep} {self.control_state[self.current_step]['target_term']}")
            max_len_reached = (self.current_substep + 1 == len(self.control_state[self.current_step]["target_term"][
                self.control_state[self.current_step]["selected_target"]]))
        elif not max_len_reached:
            max_len_reached = (self.current_substep + 1 == len(self.control_state[self.current_step]["target_term"]))

        if max_len_reached:
            self.current_step_done = True
            self.current_step += 1
            self.current_substep = 0
            print(f"----- incrementing step max_len_reached, {self.current_step}: {self.current_substep}")

            if self.current_step >= len(self.control_state):
                self.all_steps_done = True
            return True

        self.current_substep += 1
        self.control_state[key_step]["sub_step_number"] += 1
        self.control_state[key_step]["tokenized_subterm"] = tokenized_subterm
        return True

    def bias_logits(self, logits, eos_tokens=None):
        print(f"\n-------- step / substep {self.current_step} / {self.current_substep}")
        print("current control state: ", self.control_state[self.current_step])
        # Selection from options
        if self.control_state[self.current_step]["type"] == "select":
            # init option id to -1 to catch errors if not assigned to any option
            option_num = -1

            # TODO: Add in tree based search here later
            # How does guidance do it?

            # For now, checking if we match the first token as prefix and continue from there
            if self.control_state[self.current_step]["sub_step_number"] == 0:
                max_prob = -1
                next_tok_id = None
                for j, option in enumerate(self.control_state[self.current_step]["target_term"]):
                    print("****************: ", logits.shape, option[0], logits[:, option[0]])

                    max_prob = max(logits[:, option[0]], max_prob)
                    if max_prob == logits[:, option[0]]:
                        next_tok_id = option[0]
                        # option_num = j
                        self.control_state[self.current_step]["selected_target"] = j
                        self.control_state[self.current_step]["tokenized_subterm"] = next_tok_id

                print("prev logits score: ", logits[:, next_tok_id], "next_tok_id: ", next_tok_id)
                logits[:, next_tok_id] = 1000
            else:
                option_num = self.control_state[self.current_step]["selected_target"]
                next_tok_id = self.control_state[self.current_step]["target_term"][option_num][self.current_substep]
                print("prev logits score: ", logits[:, next_tok_id], "next_tok_id: ", next_tok_id)
                logits[:, next_tok_id] = 1000

            # increment the current step
            self.set_next_step(self.current_step, next_tok_id)
            print(f"----- end of increment {self.current_step}  / {self.current_substep}: {self.control_state}\n")

            # get the eos
            # This might not be needed
            # if (
            #     self.control_state[self.current_step]["sub_step_number"]
            #     == len(self.control_state[self.current_step]["target_term"][option_num])):
            #     self.current_step += 1
            return logits

        # Keys and other values
        else:
            # Keys
            if eos_tokens is None:
                next_tok_id = self.control_state[self.current_step]["target_term"][self.current_substep]
                print("prev logits score: ", logits[:, next_tok_id], "next_tok_id: ", next_tok_id)
                logits[:, next_tok_id] = 1000
                self.set_next_step(self.current_step, next_tok_id)
            # Normal values
            # This might not be needed
            else:
                max_len_reached = False
                next_tok_id = torch.argmax(logits, dim=-1).item()
                print("###### argmax check: ", next_tok_id, eos_tokens)
                if next_tok_id in eos_tokens:
                    print("stopping value", next_tok_id)
                    max_len_reached = True
                    # self.current_step += 1
                    # self.current_substep = 0
                self.set_next_step(self.current_step, next_tok_id, max_len_reached=max_len_reached)
        return logits

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Assuming no beam search. So seqs is a list of one sequence.
        # seq = seqs[0]
        self.current_step_done = False
        # self.all_steps_done = False
        substep_number = self.control_state[self.current_step]["sub_step_number"]

        # for keys and unguided values where we have the eos token set
        # if self.control_state[self.current_step]["end_token"] == torch.argmax(logits, dim=-1):
        #     print("\n-----\ncondition for keys and unguided values")
        #     self.current_step_done = True
        #     self.current_step += 1
        # keys
        if self.control_state[self.current_step]["type"] == "key":
            print("\n############################\ncondition for keys")
            # print("logits: ", logits)
            logits = self.bias_logits(logits)
        elif self.control_state[self.current_step]["type"] == "value":
            print("\n############################\ncondition for values")
            # print("logits: ", logits)
            logits = self.bias_logits(logits, self.control_state[self.current_step]["end_token"])
        # unguided value and selection value
        else:
            print("\n############################\ncondition for guided values")
            logits = self.bias_logits(logits)
        return logits
    
    def __repr__(self) -> str:
        return json.dumps(
            {
                "type": self.type,
                "schema": self.schema,
                "control_state": self.control_state,
            })

    # @classmethod
    # def from_output_control_args(cls) -> "OutputControlParams":
    #     obj = cls()
    #     return obj
