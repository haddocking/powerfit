# PyScript Web Worker for PowerFit computation
# This worker needs the same packages as the main thread

import js
from pathlib import Path
# from pyscript import fetch, window
from functools import partial

# Create a dummy progress function for PowerFit
def dummy_progress(x):
    return x

progress = partial(dummy_progress)

async def run_powerfit_computation(data):
    """Run PowerFit computation in web worker"""

    try:
        # Extract parameters from data
        # target_file = data["target_file"]
        # template_file = data["template_file"]
        # parameters = data["parameters"]
        
        print("Received data in worker:", data)  # Debug info
        return "noooo"
        return {
                "success": False,
                "error": "ended early"
            }

        # # Save files to filesystem
        # target_url = window.URL.createObjectURL(target_file)
        # target_fn = f'./{target_file.name}'
        # with open(target_fn, 'wb') as d:
        #     d.write(await fetch(target_url).bytearray())
        # window.URL.revokeObjectURL(target_url)

        # template_url = window.URL.createObjectURL(template_file)
        # template_fn = f'./{template_file.name}'
        # with open(template_fn, 'wb') as d:
        #     d.write(await fetch(template_url).bytearray())
        # window.URL.revokeObjectURL(template_url)

        # # Import PowerFit
        # from powerfit_em.powerfit import powerfit
        
        # # Run PowerFit analysis
        # with open(template_fn, 'r') as template_io, open(target_fn, 'rb') as target_io:
        #     print("Starting PowerFit in worker...")  # Debug info
        #     powerfit(
        #         target_volume=target_io,
        #         resolution=parameters["resolution"],
        #         template_structure=template_io,
        #         angle=parameters["angle"],
        #         laplace=parameters["laplace"],
        #         core_weighted=parameters["core_weighted"],
        #         no_resampling=parameters["no_resampling"],
        #         resampling_rate=parameters["resampling_rate"],
        #         no_trimming=parameters["no_trimming"],
        #         trimming_cutoff=parameters["trimming_cutoff"],
        #         chain=parameters["chain"],
        #         directory=".",
        #         num=parameters["num_models"],
        #         gpu=None,  # GPU not supported in PyScript
        #         nproc=1,   # Single-threaded in PyScript
        #         delimiter=',',
        #         progress=progress
        #     )
        # print("PowerFit analysis completed in worker.")  # Debug info

        # # Read the solutions.out file
        # solutions_path = Path("solutions.out")
        # if solutions_path.exists():
        #     solutions_content = solutions_path.read_text()
        #     return {
        #         "success": True,
        #         "solutions_content": solutions_content,
        #         "message": "PowerFit analysis completed successfully!"
        #     }
        # else:
        #     return {
        #         "success": False,
        #         "error": "Analysis completed but solutions file not found."
        #     }
            
    except Exception as e:
        print(f"PowerFit error in worker: {e}")  # Debug info
        return {
            "success": False,
            "error": str(e)
        }

# Register the worker function
js.self.run_powerfit_computation = run_powerfit_computation
