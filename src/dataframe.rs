use std::collections::HashMap;
use std::fs;
use serde::{Serialize, Deserialize, Serializer, ser::{SerializeTuple, SerializeMap}};

use rayon::prelude::*;

#[derive(Deserialize, Clone, Debug)]
pub struct Sample {
	pub mean: f32,
	pub std: f32,
	pub num_samples: usize,
}

impl Sample {
	pub fn new(s: f32) -> Sample {
		return Sample { mean: s, std: 0., num_samples: 1 };
	}

	pub fn empty() -> Sample {
		Sample { mean: 0., std: 0., num_samples: 0 }
	}

	pub fn from(vec: &Vec<f32>) -> Sample {
		let num_samples: usize = vec.len();
		let mut s: f32 = 0.;
		let mut s2: f32 = 0.;

		for v in vec {
			s += v;
			s2 += v*v;
		}
	
		s /= num_samples as f32;
		s2 /= num_samples as f32;

		let mean: f32 = s;
		let std: f32 = (s2 - s*s).powf(0.5);

		return Sample { mean: mean, std: std, num_samples: num_samples }
	}

	pub fn collapse(samples: Vec<Sample>) -> Sample {
		samples.iter().fold(Sample { mean: 0., std: 0., num_samples: 0 }, |sum, val| sum.combine(val))
	}

	pub fn combine(&self, other: &Sample) -> Sample {
		let combined_samples: usize = self.num_samples + other.num_samples;
		let combined_mean: f32 = ((self.num_samples as f32)*self.mean
								+ (other.num_samples as f32)*other.mean)
								/ (combined_samples as f32);
		let combined_std: f32 = (((self.num_samples as f32)* (self.std.powi(2) +  (self.mean -  combined_mean).powi(2)) 
							    + (other.num_samples as f32)*(other.std.powi(2) + (other.mean - combined_mean).powi(2)))
							    / (combined_samples as f32)).powf(0.5);
		return Sample { mean: combined_mean, std: combined_std, num_samples: combined_samples };
	}
}

impl Serialize for Sample {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_tuple(3)?;
		seq.serialize_element(&self.mean)?;
		seq.serialize_element(&self.std)?;
		seq.serialize_element(&self.num_samples)?;
		seq.end()
    }
}

// Code for managing output data in runs
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DataField {
	Int(i32),
	Float(f32),
	String(String),
	Data(Vec<Sample>),
}

impl DataField {
	pub fn congruent(&self, other: &DataField) -> bool {
		match self {
			DataField::Int(x) => {
				match other {
					DataField::Int(y) => return x == y,
					_ => return false
				}
			},
			DataField::Float(x) => {
				match other {
					// Note: will be buggy for small Floats
					DataField::Float(y) => return (x - y).abs() < 0.001,
					_ => return false
				}
			},
			DataField::String(x) => {
				match other {
					DataField::String(y) => return x == y,
					_ => return false
				}
			},
			DataField::Data(v) => {
				match other {
					DataField::Data(w) => return v.len() == w.len(),
					_ => return false
				}
			}
		}
	}
}

#[derive(Serialize, Deserialize)]
pub struct DataFrame {
	pub params: HashMap<String, DataField>,
	pub slides: Vec<DataSlide>
}

impl DataFrame {
	pub fn new() -> Self {
		return DataFrame { params: HashMap::new(), slides: Vec::new() };
	}

	pub fn from(mut slides: Vec<DataSlide>) -> Self {
		let mut df = DataFrame::new();
		for i in 0..slides.len() {
			df.add_slide(slides.pop().unwrap());
		}
		return df;
	}

	pub fn add_int_param(&mut self, key: &str, val: i32) {
		self.params.insert(String::from(key), DataField::Int(val));
	}

	pub fn add_float_param(&mut self, key: &str, val: f32) {
		self.params.insert(String::from(key), DataField::Float(val));
	}

	pub fn add_string_param(&mut self, key: &str, val: String) {
		self.params.insert(String::from(key), DataField::String(val));
	}


	pub fn add_slide(&mut self, slide: DataSlide) {
		self.slides.push(slide);
	}

	pub fn save_json(&self, filename: String) {
		let json = serde_json::to_string_pretty(&self).unwrap();
		fs::write(filename, json);
	}
}

#[derive(Deserialize, Clone, Debug)]
pub struct DataSlide {
	data: HashMap<String, DataField>,
}

impl Serialize for DataSlide {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_map(Option::Some(self.data.len()))?;
		for (key, val) in &self.data {
			seq.serialize_entry(key, val);
		}

		seq.end()
    }
}

impl DataSlide {
	pub fn new() -> DataSlide {
		let data: HashMap<String, DataField> = HashMap::new(); 
		return DataSlide { data: data };
	}

	pub fn add_int_param(&mut self, key: &str, val: i32) {
		self.data.insert(String::from(key), DataField::Int(val));
	}

	pub fn add_float_param(&mut self, key: &str, val: f32) {
		self.data.insert(String::from(key), DataField::Float(val));
	}

	pub fn add_string_param(&mut self, key: &str, val: String) {
		self.data.insert(String::from(key), DataField::String(val));
	}

	pub fn push_data(&mut self, key: &str, val: Sample) {
		match self.data.get_mut(key).unwrap() {
			DataField::Data(v) => v.push(val),
			_ => ()
		}
	}

	pub fn congruent(&self, other: &DataSlide) -> bool {
		for key in self.data.keys() {
			if !other.data.contains_key(key) { return false }
		}

		for key in other.data.keys() {
			if !self.data.contains_key(key) { return false }
		}

		for (key, datafield) in &self.data {
			if other.data.contains_key(key) {
				if !datafield.congruent(&other.get_val(&key)) { return false }
			}
		}

		true
	}

	pub fn add_data(&mut self, key: &str) {
		self.data.insert(String::from(key), DataField::Data(Vec::new()));
	}

	pub fn get_val(&self, key: &str) -> &DataField {
		return &self.data[key];
	}

	pub fn contains_key(&self, key: &str) -> bool {
		return self.data.contains_key(key);
	}

	pub fn unwrap_int(&self, key: &str) -> i32 {
		match self.data[key] {
			DataField::Int(x) => x,
			_ => panic!()
		}
	}

	pub fn unwrap_float(&self, key: &str) -> f32 {
		match self.data[key] {
			DataField::Float(x) => x,
			_ => panic!()
		}
	}

	pub fn unwrap_string(&self, key: &str) -> String {
		match &self.data[key] {
			DataField::String(x) => x.clone(),
			_ => panic!()
		}
	}

	pub fn unwrap_data(&self, key: &str) -> &Vec<Sample> {
		match &self.data[key] {
			DataField::Data(x) => x,
			_ => panic!()
		}
	}

	pub fn combine(&self, other: &DataSlide) -> DataSlide {
		let mut dataslide: DataSlide = DataSlide::new();
		assert!(self.congruent(other));

		for (key, datafield) in &self.data {
			match datafield {
				DataField::Int(x) => dataslide.add_int_param(&key, *x),
				DataField::Float(x) => dataslide.add_float_param(&key, *x),
				DataField::String(x) => dataslide.add_string_param(&key, x.clone()),
				DataField::Data(x) => {
					match &other.data[key] {
						DataField::Data(y) => {
							dataslide.add_data(&key);
							for i in 0..x.len() {
								dataslide.push_data(&key, x[i].combine(&y[i]));
							}
						},
						_ => {
							println!("DataSlides cannot be combined.");
							panic!();
						}
					}
				}
			}
		}

		dataslide
	}
}




// Code for managing parallel computation of many configurable runs
pub trait RunConfig {
    fn init_state(&mut self);
    fn gen_dataslide(&mut self) -> DataSlide;
}

pub struct ParallelCompute<C> {
    num_threads: usize,
    configs: Vec<C>,
	params: HashMap<String, DataField>,

    initialized: bool,
}

impl<C: RunConfig + std::marker::Sync + std::marker::Send + Clone> ParallelCompute<C> {
    pub fn new(num_threads: usize, configs: Vec<C>) -> Self {
        Self { num_threads: num_threads, configs: configs, params: HashMap::new(), initialized: false }
    }

	pub fn add_int_param(&mut self, key: &str, val: i32) {
		self.params.insert(String::from(key), DataField::Int(val));
	}

	pub fn add_float_param(&mut self, key: &str, val: f32) {
		self.params.insert(String::from(key), DataField::Float(val));
	}

	pub fn add_string_param(&mut self, key: &str, val: String) {
		self.params.insert(String::from(key), DataField::String(val));
	}

    pub fn compute(&mut self) -> DataFrame {
        if !self.initialized {
            rayon::ThreadPoolBuilder::new().num_threads(self.num_threads).build_global().unwrap();
        }

        let slides: Vec<DataSlide> = (0..self.configs.len()).into_par_iter().map(|i| {
			// TODO avoid cloning configs
			let mut config = self.configs[i].clone();
			config.init_state();
        	config.gen_dataslide()
        }).collect();


		let mut avg_slides: Vec<DataSlide> = Vec::new();

		for slide1 in &slides {
			// Integrate with avg_slides
			let mut idx: usize = 0;
			let mut found_congruent: bool = false;
			for (i, slide2) in avg_slides.iter().enumerate() {
				if slide1.congruent(slide2) {
					found_congruent = true;
					idx = i;
					break;
				}
			}
			if found_congruent {
				avg_slides[idx] = avg_slides[idx].combine(slide1);
			} else {
				avg_slides.push(slide1.clone());
			}
		}

		DataFrame { params: self.params.clone(), slides: avg_slides }
    }
}