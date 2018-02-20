/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  num_particles = 200;

  for (decltype(num_particles) i = 0; i != num_particles; i++) {
    struct Particle sample_particle;
    sample_particle.id = i;
    sample_particle.x = dist_x(gen);
    sample_particle.y = dist_y(gen);
    sample_particle.theta = dist_theta(gen);
    sample_particle.weight = 1;
    weights.push_back(1);
    particles.push_back(sample_particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  for (auto &particle : particles) {
    double &x = particle.x;
    double &y = particle.y;
    double &theta = particle.theta;

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    if (fabs(yaw_rate) > 0.000001) {
      x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      theta += yaw_rate * delta_t;
    }
    else {
      x += velocity * delta_t * cos(theta);
      y += velocity * delta_t * sin(theta);
    }

    // Add random Gaussian noise
    x += dist_x(gen);
    y += dist_y(gen);
    theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (decltype(observations.size()) i = 0; i != observations.size(); i++) {
    double nearest_distance = std::numeric_limits<double>::max();
    int nearest_pred_idx = -1;
    double xo = observations[i].x;
    double yo = observations[i].y;

    for (decltype(predicted.size()) j = 0; j != predicted.size(); j++) {
      double xp = predicted[j].x;
      double yp = predicted[j].y;
      double distance = dist(xp, yp, xo, yo);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_pred_idx = j;
      }
    }
    if (nearest_pred_idx != -1) {
      observations[i].id = predicted[nearest_pred_idx].id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));


  for (decltype(particles.size()) i = 0; i != particles.size(); i++) {
    double theta = particles[i].theta;
    double part_x = particles[i].x;
    double part_y = particles[i].y;

    // Step1: Transform observations
    vector<LandmarkObs> trans_observations;
    for (decltype(observations.size()) j = 0; j != observations.size(); j++) { 
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;

      double map_x = part_x + cos(theta) * obs_x - sin(theta) * obs_y;
      double map_y = part_y + sin(theta) * obs_x + cos(theta) * obs_y;
      trans_observations.push_back(LandmarkObs{ observations[i].id, map_x, map_y });
    }

    // Step2: Find landmarks in sense range
    vector<LandmarkObs> predicted;
    for (decltype(map_landmarks.landmark_list.size()) j = 0; j != map_landmarks.landmark_list.size(); j++) {
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      if (dist(part_x, part_y, landmark_x, landmark_y) <= sensor_range) {
        predicted.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
    }

    // Step3: Association
    dataAssociation(predicted, trans_observations);

    // Step4: Calculate particle's weight
    for (auto trans_o : trans_observations) {
      int id = trans_o.id;
      double obs_x = trans_o.x;
      double obs_y = trans_o.y;
      for (auto pred : predicted) {
        if (pred.id == id) {
          double mu_x = pred.x;
          double mu_y = pred.y;
          double exponent = pow((obs_x - mu_x), 2) / (2 * pow(sig_x, 2)) + pow((obs_y - mu_y), 2) / (2 * pow(sig_y, 2));
          particles[i].weight *= gauss_norm * exp(-exponent);
        }
      }
      weights[i] = particles[i].weight;
    } // for (auto trans_o : trans_observations)
  } // for (decltype(observations.size()) j = 0; j != observations.size(); j++)
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> resample_particles;

  for (decltype(num_particles) i = 0; i != num_particles; i++) {
    resample_particles.push_back(particles[d(gen)]);
  }
  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
