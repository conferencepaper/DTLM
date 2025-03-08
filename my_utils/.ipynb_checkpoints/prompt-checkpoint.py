def create_prompt(list_input, description, example_input_list, example_output_list, input_output_structure="default", with_output=False):
    # Template for the prompt
    prompt_template = """
You are a direct data transformation tool. 
Given a list of values in different data formats and a target format, 
the objective is to transform individual values to align with the new format.
The model should transform each provided value with 100% accuracy. 
If the model is unsure about any transformation, it should return the original value. The number of transformed
values in the output should match the number of input values.
Description of the transformation: {description}
"""

    # Initialize empty input-output example string
    input_output_examples = ""

    if example_input_list:
        
        # Default structure for input-output
        if input_output_structure == "default":
            input_output_examples = f"Inputs: {example_input_list}\nOutput: {example_output_list}"

        # One input-output pair per line
        elif input_output_structure == "one":
            input_output_examples = ""
            for i, (inp, out) in enumerate(zip(example_input_list, example_output_list)):
                input_output_examples += f"Input: [{inp}]\nOutput: [{out}]\n"
    
        # Compact structure
        elif input_output_structure == "compact":
            input_output_examples = ", ".join([f"[{inp}] -> [{out}]" for inp, out in zip(example_input_list, example_output_list)])
    
        # Bullet Point structure
        elif input_output_structure == "bullet":
            input_output_examples = "\n".join([f"- Input: {inp} -> Output: {out}" for inp, out in zip(example_input_list, example_output_list)])
    
        # Step-by-Step structure
        elif input_output_structure == "step_by_step":
            input_output_examples = "\n".join([f"Step {i+1}: Transform {inp} -> {out}" for i, (inp, out) in enumerate(zip(example_input_list, example_output_list))])
        input_output_examples="\n"+input_output_examples
    else:
        input_output_examples = ""

    # Combine the prompt template with the description and examples
    full_prompt = prompt_template.format(description=description) +input_output_examples + "\n" + "Inputs: " + str(list_input)

    if with_output:
        full_prompt += "\n" + "Output:"
    
    return full_prompt

"""
# Example Usage
description = "Transform lowercase letters to uppercase."
example_input_list = ["a","b","c"]
example_output_list = ['X', 'Y', 'Z']
list_input = ["skander", "zzzzz", "aaaaa"]

# Generate a prompt with default structure
print(create_prompt(list_input, description, example_input_list, example_output_list, input_output_structure="default", with_output=True))

# Generate a prompt with bullet point structure
print(create_prompt(list_input, description, example_input_list, example_output_list, input_output_structure="bullet"))
print(create_prompt(list_input, description, example_input_list, example_output_list, input_output_structure="compact"))
print(create_prompt(list_input, description, example_input_list, example_output_list, input_output_structure="step_by_step"))
print(create_prompt(list_input, description, example_input_list, example_output_list, input_output_structure="one"))"""
