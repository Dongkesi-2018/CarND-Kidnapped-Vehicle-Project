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

  num_particles = 400;

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

  for (decltype(num_particles) i = 0; i != num_particles; i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    double x_f = 0, y_f = 0, theta_f = 0;

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    if (fabs(yaw_rate) > 0.000001) {
      x_f = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      y_f = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      theta_f = theta + yaw_rate * delta_t;
    }
    else {
      x_f = x + velocity * delta_t * cos(theta);
      y_f = y + velocity * delta_t * sin(theta);
      theta_f = theta;
    }

    // Add random Gaussian noise
    x_f += dist_x(gen);
    y_f += dist_y(gen);
    theta_f += dist_theta(gen);

    particles[i].x = x_f;
    particles[i].y = y_f;
    particles[i].theta = theta_f;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (auto i = 0; i != predicted.size(); i++) {
    double nearest_distance = std::numeric_limits<double>::max();
    int nearest_obs_idx = -1;
    double xp = predicted[i].x;
    double yp = predicted[i].y;
    for (auto j = 0; j != observations.size(); j++) {
      double xo = observations[j].x;
      double yo = observations[j].y;
      double distance = dist(xp, yp, xo, yo);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_obs_idx = j;
      }
    }
    if (nearest_obs_idx != -1) {
      // associate predicted[i] with observations[nearest_obs_idx]
      predicted[i].id = observations[nearest_obs_idx].id;
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
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    for (decltype(observations.size()) j = 0; j != observations.size(); j++) {
      // Step1: Transform
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;
      double theta = particles[i].theta;
      double part_x = particles[i].x;
      double part_y = particles[i].y;

      double map_x = part_x + cos(theta) * obs_x - sin(theta) * obs_y;
      double map_y = part_y + sin(theta) * obs_x + cos(theta) * obs_y;

      // Step2: Find the nearest map landmark
      int nearest_landmark_idx = -1;
      double nearest_distance = sensor_range;
      for (decltype(map_landmarks.landmark_list.size()) k = 0; k != map_landmarks.landmark_list.size(); k++) {
        double landmark_x = map_landmarks.landmark_list[k].x_f;
        double landmark_y = map_landmarks.landmark_list[k].y_f;
        double distance = dist(map_x, map_y, landmark_x, landmark_y);
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest_landmark_idx = k;
        }
      }

      // Step3: Calculate particle's weight
      if (nearest_landmark_idx != -1 && nearest_distance <= sensor_range) {
        double mu_x = map_landmarks.landmark_list[nearest_landmark_idx].x_f;
        double mu_y = map_landmarks.landmark_list[nearest_landmark_idx].y_f;

        double exponent = pow((obs_x - mu_x), 2) / (2 * pow(sig_x, 2)) + pow((obs_y - mu_y), 2) / (2 * pow(sig_y, 2));
        particles[i].weight *= gauss_norm * exp(-exponent);

        int id = map_landmarks.landmark_list[nearest_landmark_idx].id_i;
        associations.push_back(id);
        sense_x.push_back(mu_x);
        sense_y.push_back(mu_y);
      }
    }
    SetAssociations(particles[i], associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  std::vector<double> weights;
  for (decltype(particles.size()) i = 0; i != particles.size(); i++) {
    weights.push_back(particles[i].weight);
  }

  std::size_t i(0);
  discrete_distribution<> dist(weights.size(), 0, 0, [&weights, &i](double) {
    return weights[i++];
  });
  vector<Particle> resample_particles;

  for (decltype(num_particles) i = 0; i != num_particles; i++) {
    resample_particles.push_back(particles[dist(gen)]);
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
