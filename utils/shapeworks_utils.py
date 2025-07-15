import shapeworks as sw
import os




def make_project_files(meshes,particles,data_dir,proj_name):
	# Create project spreadsheet
	project_location =  data_dir + "shape_models/"
	if not os.path.exists(project_location):
		os.makedirs(project_location)
	# Set subjects
	subjects = []
	print(len(meshes),len(particles))
	number_domains = 1
	for i in range(len(particles)):
		subject = sw.Subject()
		subject.set_number_of_domains(number_domains)
		rel_seg_files =sw.utils.get_relative_paths([meshes[i]], project_location)
		subject.set_original_filenames(rel_seg_files)
		subject.set_groomed_filenames(rel_seg_files)
		f =sw.utils.get_relative_paths([particles[i]], project_location)
		subject.set_local_particle_filenames(f)
		subject.set_world_particle_filenames(f)
		subjects.append(subject)
	project = sw.Project()
	project.set_subjects(subjects)
	spreadsheet_file = project_location + proj_name +".xlsx"
	print(spreadsheet_file)
	project.save(spreadsheet_file)
	return spreadsheet_file


def get_stats(particles_list):
	particleSystem = sw.ParticleSystem(particles_list)
	print("Calculating Compactness")
	allCompactness = sw.ShapeEvaluation.ComputeFullCompactness(particleSystem=particleSystem)
	print("Calculating Generalization")
	allGeneralization = sw.ShapeEvaluation.ComputeFullGeneralization(particleSystem=particleSystem)
	print("Calculating Specificity")
	allSpecificity = sw.ShapeEvaluation.ComputeFullSpecificity(particleSystem=particleSystem)
	return allCompactness, allGeneralization, allSpecificity