#pragma once

std::vector<float> LoadStandardDev(std::string fileName,bool header);

std::vector<std::vector<float>> LoadRotation(std::string fileName, bool header);

std::vector<float> LoadTrain(std::string fileName, bool header);

std::vector<float> LoadAverageVector(std::string fileName, bool header);