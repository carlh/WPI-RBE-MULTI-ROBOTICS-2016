#include <iostream>
#include <sstream>
#include <thread>
#include <mutex>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <sys/socket.h>

#include "rbe510.hpp"

#include <fcntl.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <json/json.h>

using namespace std;
using namespace cv;

#define ENTITY_STARTING_INDEX 100
#define CORNER_STARTING_INDEX 200
#define PI 3.1415926535

#define TOP_LEFT 200
#define BOTTOM_LEFT 201
#define TOP_RIGHT 202
#define BOTTOM_RIGHT 203

/* Mutex for the data sanity */
std::mutex mtx;
std::mutex broadcast_mtx;

vector<Robot> robots;
vector<Entity> entities;

bool _drawGrid = true;
int _gridSpacing = 100;

class BT {
public:
	vector<int> files;
	vector<string> message_queue;

	BT(){
		thread update_thread(&BT::thread_program, this);
		update_thread.detach();
	}

	void open(string port){
		int fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
		if(fd == -1){
			cout << "Unable to open " << port << endl;
		}else{
			files.push_back(fd);
			set_interface_attribs(fd, B115200, 0);
			set_blocking(fd, 0);
		}
	}

	void broadcastToAll(string message){
		for(unsigned i = 0; i < files.size(); i++){
			write(files[i], message.c_str(), message.size());
		}
	}

	void queue(string message){
		broadcast_mtx.lock();
		message_queue.push_back(message);
		broadcast_mtx.unlock();
	}

	void thread_program(){
		while(1){
			if (!message_queue.empty()) {
				broadcast_mtx.lock();
				broadcastToAll(message_queue.front());
				message_queue.erase(message_queue.begin());
				broadcast_mtx.unlock();
			}
		}
	}

	int set_interface_attribs (int fd, int speed, int parity){
        struct termios tty;
        memset (&tty, 0, sizeof tty);
        cfsetospeed (&tty, speed);
        cfsetispeed (&tty, speed);
        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
        // disable IGNBRK for mismatched speed tests; otherwise receive break
        // as \000 chars
        tty.c_iflag &= ~IGNBRK;         // disable break processing
        tty.c_lflag = 0;                // no signaling chars, no echo,
                                        // no canonical processing
        tty.c_oflag = 0;                // no remapping, no delays
        tty.c_cc[VMIN]  = 0;            // read doesn't block
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout
        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl
        tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
                                        // enable reading
        tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
        tty.c_cflag |= parity;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;
        return 0;
	}

	void set_blocking (int fd, int should_block){
        struct termios tty;
        memset (&tty, 0, sizeof tty);
        tty.c_cc[VMIN]  = should_block ? 1 : 0;
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout
	}
};

BT blue;

struct Field {
private:
    Entity _topLeft;
    Entity _topRight;
    Entity _bottomRight;
    Entity _bottomLeft;

public:
    Point2f TopLeftCoord() { return Point2f(_topLeft.x(), _topLeft.y()); }
    void SetTopLeft(Entity topLeft) { _topLeft = topLeft; }

    Point2f TopRightCoord() { return Point2f(_topRight.x(), _topRight.y()); }
    void SetTopRight(Entity topRight) { _topRight = topRight; }

    Point2f BottomRightCoord() { return Point2f(_bottomRight.x(), _bottomRight.y()); }
    void SetBottomRight(Entity bottomRight) { _bottomRight = bottomRight; }

    Point2f BottomLeftCoord() { return Point2f(_bottomLeft.x(), _bottomLeft.y()); }
    void SetBottomLeft(Entity bottomLeft) { _bottomLeft = bottomLeft; }

    void CreateField(Entity topLeft, Entity topRight, Entity bottomRight, Entity bottomLeft) {
        _topLeft = topLeft;
        _topRight = topRight;
        _bottomLeft = bottomLeft;
        _bottomRight = bottomRight;
    }

    cv::Size Size() {
        int width = static_cast<int>(_topRight.x() + _topRight.width() - _topLeft.x());
        int height = static_cast<int>(_bottomLeft.y() - _topLeft.y());
        return cv::Size(width, height);
    }

    bool IsValid() {
        return !(_topLeft.id() == -1
                 || _topRight.id() == -1
                 || _bottomRight.id() == -1
                 || _bottomLeft.id() == -1);
    }

    string ToJson() {
        Json::Value data = Json::objectValue;
        data["ready"] = IsValid();
        data["topLeft"]["coordinates"]["x"] = _topLeft.x();
        data["topLeft"]["coordinates"]["y"] = _topLeft.y();
        data["topRight"]["coordinates"]["x"] = _topRight.x();
        data["topRight"]["coordinates"]["y"] = _topRight.y();
        data["bottomRight"]["coordinates"]["x"] = _bottomRight.x();
        data["bottomRight"]["coordinates"]["y"] = _bottomRight.y();
        data["bottomLeft"]["coordinates"]["x"] = _bottomLeft.x();
        data["bottomLeft"]["coordinates"]["y"] = _bottomLeft.y();
        data["size"]["width"] = Size().width;
        data["size"]["height"] = Size().height;
        return data.toStyledString();
    }

public:
    Field() {}
};

Field _field;

void updateField() {
    for (auto entity : entities) {
        if (entity.id() == TOP_LEFT) {
            _field.SetTopLeft(entity);
        } else if (entity.id() == TOP_RIGHT) {
            _field.SetTopRight(entity);
        } else if (entity.id() == BOTTOM_RIGHT) {
            _field.SetBottomRight(entity);
        } else if (entity.id() == BOTTOM_LEFT) {
            _field.SetBottomLeft(entity);
        }
    }
}

/* Finds distance between two points */
float dist2pf(Point2f a, Point2f b){ return sqrt(pow(a.x - b.x,2.0) + pow(a.y -  b.y,2.0)); }

/* Finds the middle point between two points */
Point2f midPoint(Point2f a, Point2f b){ return Point2f((a.x + b.x)/2.0,(a.y + b.y)/2.0); }

/* Finds the center of a square */
Point2f findSquareCenter(vector<Point2f> square){
	Point2f mid1 = midPoint(square[0], square[1]);
	Point2f mid3 = midPoint(square[2], square[3]);
	return midPoint(mid1, mid3);
}

/* Runs the Aruco program */
void runAruco(Mat image, vector<int> &markerIds, vector< vector<Point2f> > &markerCorners){
	if(image.empty()) return;
	vector<vector<Point2f> > rejectedCandidates;
	cv::Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
	cv::Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
	aruco::detectMarkers(image, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
}

void createInversePerspectiveTransform(Mat& perspectiveTransform) {
    if (!_field.IsValid()) {
        return;
    }

    Size fieldSize = _field.Size();
    vector<Point2f> destCorners;
    destCorners.push_back(_field.TopLeftCoord());
    destCorners.push_back(_field.TopRightCoord());
    destCorners.push_back(_field.BottomRightCoord());
    destCorners.push_back(_field.BottomLeftCoord());

    vector<Point2f>sourceCorners;
    sourceCorners.push_back(Point2f(0, 0));
    sourceCorners.push_back(Point2f(fieldSize.width, 0));
    sourceCorners.push_back(Point2f(fieldSize.width, fieldSize.height));
    sourceCorners.push_back(Point2f(0, fieldSize.height));

    perspectiveTransform = getPerspectiveTransform(sourceCorners, destCorners);
}

void createPerspectiveTransform(Mat& perspectiveTransform) {
    if (!_field.IsValid()) {
        return;
    }

    Size fieldSize = _field.Size();
    vector<Point2f> sourceCorners;
    sourceCorners.push_back(_field.TopLeftCoord());
    sourceCorners.push_back(_field.TopRightCoord());
    sourceCorners.push_back(_field.BottomRightCoord());
    sourceCorners.push_back(_field.BottomLeftCoord());

    vector<Point2f>destCorners;
    destCorners.push_back(Point2f(0, 0));
    destCorners.push_back(Point2f(fieldSize.width, 0));
    destCorners.push_back(Point2f(fieldSize.width, fieldSize.height));
    destCorners.push_back(Point2f(0, fieldSize.height));

    perspectiveTransform = getPerspectiveTransform(sourceCorners, destCorners);
}

void applyPerspectiveToObjects() {
    Mat perspectiveTransform;
    createPerspectiveTransform(perspectiveTransform);

}

void applyPerspective(Mat& image) {
    if (!_field.IsValid()) {
        return;
    }
    Mat perspectiveXfrm;
    createPerspectiveTransform(perspectiveXfrm);

    warpPerspective(image, image, perspectiveXfrm, _field.Size());

    vector<Point2f> entityCoords;

    entityCoords.push_back(_field.TopLeftCoord());
    entityCoords.push_back(_field.TopRightCoord());
    entityCoords.push_back(_field.BottomRightCoord());
    entityCoords.push_back(_field.BottomLeftCoord());

    perspectiveTransform(entityCoords, entityCoords, perspectiveXfrm);

    for (auto coord : entityCoords) {
        circle(image, coord, 10, Scalar(10, 50, 255));
    }
}

void perspectiveTransformPoint(Point2f& pt, Mat transformationMatrix) {
    vector<Point2f> pts {pt};
    perspectiveTransform(pts, pts, transformationMatrix);
    pt = pts[0];
}

void perspectiveTransformGridToPixel(Point2f& pt) {
    Mat transformMatrix;
    createPerspectiveTransform(transformMatrix);
    perspectiveTransformPoint(pt, transformMatrix);
}

void perspectiveTransformPixelToGrid(Point2f& pt) {
    Mat transformMatrix;
    createInversePerspectiveTransform(transformMatrix);
    perspectiveTransformPoint(pt, transformMatrix);
}

void transformEntities(vector<Robot>& robots, vector<Entity>& entities) {
    Mat transformMatrix;
    createPerspectiveTransform(transformMatrix);
    vector<Point2f> pts;

    for (int i = 0; i < entities.size(); i++) {
        Point2f pt(entities[i].x(), entities[i].y());
        pts.push_back(pt);
    }
    perspectiveTransform(pts, pts, transformMatrix);
    for (int i = 0; i < entities.size(); i++) {
		entities[i].set_x(pts[i].x);
		entities[i].set_y(pts[i].y);
    }
	pts.clear();
	for (int i = 0; i < robots.size(); i++) {
		Point2f pt(robots[i].x(), robots[i].y());
		pts.push_back(pt);
	}
	perspectiveTransform(pts, pts, transformMatrix);
	for (int i = 0; i < robots.size(); i++) {
		robots[i].set_x(pts[i].x);
		robots[i].set_y(pts[i].y);
	}
}

void drawGrid(Mat const& image) {
    if (!_field.IsValid()) {
        return;
    }

    Point2f topLeft = _field.TopLeftCoord();
    Point2f topRight = _field.TopRightCoord();
    Point2f bottomRight = _field.BottomRightCoord();
    Point2f bottomLeft = _field.BottomLeftCoord();

    Size fieldSize = _field.Size();
    double verticalLines = ceil(fieldSize.width / (double)_gridSpacing);
    double horizontalLines = ceil(fieldSize.height / (double)_gridSpacing);

    vector<Point2f> pts;
    Point2f topLeftPt(topLeft.x, topLeft.y);
    Point2f topRightPt(topRight.x, topRight.y);
    Point2f bottomLeftPt(bottomLeft.x, bottomLeft.y);
    Point2f bottomRightPt(bottomRight.x, bottomRight.y);

    pts.push_back(topLeftPt);
    pts.push_back(topRightPt);
    pts.push_back(bottomRightPt);
    pts.push_back(bottomLeftPt);


    Mat perspectiveXfrm;
    createPerspectiveTransform(perspectiveXfrm);
    perspectiveTransform(pts, pts, perspectiveXfrm);
    topLeft = pts[0];
    topRight = pts[1];
    bottomRight = pts[2];
    bottomLeft = pts[3];

    for (int i = 0; i < horizontalLines; i++) {
        Point2f startPoint(topLeft.x, topLeft.y + i * _gridSpacing);
        Point2f endPoint(topRight.x, topLeft.y + i * _gridSpacing);

        cv::line(image, startPoint, endPoint, cv::Scalar(0, 255, 255));
    }

    for (int i = 0; i < verticalLines; i++) {
        Point2f startPoint(topLeft.x + i * _gridSpacing, topLeft.y);
        Point2f endPoint(bottomLeft.x + i * _gridSpacing, bottomLeft.y);
        cv::line(image, startPoint, endPoint, cv::Scalar(255, 0, 255));
    }

}

/* Attends to the client on the network */
void attendClient(int connfd, vector<Robot> robots, vector<Entity> entities){
	socklen_t len;
	struct sockaddr_storage addr;
	char ipstr[INET6_ADDRSTRLEN];
	int port;

	len = sizeof addr;
	getpeername(connfd, (struct sockaddr*)&addr, &len);

	// deal with both IPv4 and IPv6:
	if (addr.ss_family == AF_INET) {
		struct sockaddr_in *s = (struct sockaddr_in *)&addr;
		port = ntohs(s->sin_port);
		inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);
	} else { // AF_INET6
		struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
		port = ntohs(s->sin6_port);
		inet_ntop(AF_INET6, &s->sin6_addr, ipstr, sizeof ipstr);
	}

	string receive_buffer = NetUtil::readFromSocket(connfd);

	if(receive_buffer.compare("DRIVE") == 0){
		receive_buffer = NetUtil::readFromSocket(connfd);
		stringstream parser(receive_buffer);
		int id;
		float speed;
		float sway;
		parser >> id >> speed >> sway;
		if (!parser.fail()){
			cout << ipstr << ":" << port << " requested to drive"
			<< " id: " << id
			<< " speed: " << speed
			<< " sway: " << sway
			<< endl;
		}
		stringstream deconstructor;
		deconstructor << id << " " << speed << " " << sway << "\n";
		blue.queue(deconstructor.str());
	} else if (receive_buffer.compare("DRAW_GRID") == 0) {
        cout << ipstr << ":" << port << " requested show grid" << endl;
        receive_buffer = NetUtil::readFromSocket(connfd);
        stringstream parser(receive_buffer);
        int gridSpacing;
        parser >> gridSpacing;
        cout << "Grid Spacing: " << gridSpacing;
        _drawGrid = true;
        _gridSpacing = gridSpacing > 0 ? gridSpacing : 10;
    } else if (receive_buffer.compare("HIDE_GRID") == 0) {
        cout << ipstr << ":" << port << " requested hide grid." << endl;
        _drawGrid = false;
    } else if(receive_buffer.compare("UPDATE") == 0){
		cout << ipstr << ":" << port << " requested update" << endl;
		string send_buffer;
        transformEntities(robots, entities);
		for(unsigned i = 0; i < robots.size(); i++){
			send_buffer.append(robots[i].toStr());
			send_buffer.append("\n");
		}
		for(unsigned i = 0; i < entities.size(); i++){
			send_buffer.append(entities[i].toStr());
			send_buffer.append("\n");
		}
		write(connfd, send_buffer.c_str(), send_buffer.size());
	} else if (receive_buffer.compare("FIELD_GEOMETRY") == 0) {
        cout << ipstr << ":" << port << " requested field geometry." << endl;
        string send_buffer = _field.ToJson();
        send_buffer.append("\n");
        write(connfd, send_buffer.c_str(), send_buffer.size());
    } else if (receive_buffer.compare("GRID_TO_PIXEL") == 0) {
        receive_buffer = NetUtil::readFromSocket(connfd);
        stringstream parser(receive_buffer);
        float x, y;
        parser >> x >> y;
        if (!parser.fail()) {
            Point2f pt(x, y);
            perspectiveTransformGridToPixel(pt);

            stringstream send_buffer;
            send_buffer << "PIXEL_POINT" << pt.x << " " << pt.y << " " << "\n";
            write(connfd, send_buffer.str().c_str(), send_buffer.str().size());
        }
    } else if (receive_buffer.compare("PIXEL_TO_GRID") == 0) {
        receive_buffer = NetUtil::readFromSocket(connfd);
        stringstream parser(receive_buffer);
        float x, y;
        parser >> x >> y;
        if (!parser.fail()) {
            Point2f pt(x, y);
            perspectiveTransformPixelToGrid(pt);

            stringstream send_buffer;
            send_buffer << "GRID_POINT" << pt.x << " " << pt.y << "\n";
            write(connfd, send_buffer.str().c_str(), send_buffer.str().size());
        }
    } else if(receive_buffer.compare("UPDATE_ENTITIES") == 0) {
		receive_buffer = NetUtil::readFromSocket(connfd);
		stringstream parser(receive_buffer);
		stringstream display;
		const char * separator = "";
		vector<int> updates(istream_iterator<int>(parser), {});

		display << ipstr << ":" << port << " requested update for ";
		string send_buffer;
		for (unsigned i = 0; i < updates.size(); i++) {
			int id = updates[i];
			display << separator << id;
			separator = ", ";
			if (id < ENTITY_STARTING_INDEX) {
				for (unsigned j = 0; j < robots.size(); j++) {
					if (id == robots[j].id()) {
						send_buffer.append(robots[i].toStr());
						send_buffer.append("\n");
						break;
					}
				}
			} else {
				for (unsigned j = 0; j < entities.size(); j++) {
					if (id == entities[j].id()) {
						send_buffer.append(entities[j].toStr());
						send_buffer.append("\n");
						break;
					}
				}
			}
		}
		cout << display.str() << endl;
		write(connfd, send_buffer.c_str(), send_buffer.size());
	} else if(receive_buffer.compare("GRIPPER") == 0){
		receive_buffer = NetUtil::readFromSocket(connfd);
		stringstream parser(receive_buffer);
		int id;
		float open;
		parser >> id >> open;
		if (!parser.fail()){
			cout << ipstr << ":" << port << " requested to adjust gripper"
			<< " id: " << id
			<< " open: " << open
			<< endl;
		}
		stringstream deconstructor;
		deconstructor << id << " - " << open << "\n";
		blue.queue(deconstructor.str());
	}
	close(connfd);
}

/* Network thread running seperately from the detector */
void network(vector<Robot> &robots, vector<Entity> &entities){
	int listenfd = NetUtil::getServerSocket();
	if(listenfd < 0) {
		cout << "Error: Connection problem" << endl;
		exit(1);
	}
	cout << "Network thread working. Now accepting client connections" << endl;
	while(1){
		int connfd = accept(listenfd, (struct sockaddr*)NULL, NULL);
		mtx.lock();
		std::thread pic(attendClient, connfd, robots, entities);
		pic.detach();
		mtx.unlock();
	}
}

/* Updates the database */
void update( vector<int> markerIds, vector< vector<Point2f> > markerCorners) {
	mtx.lock();
	robots.clear();
	entities.clear();
	for(unsigned i = 0; i < markerIds.size(); i++){
		Point2f center = findSquareCenter(markerCorners[i]);
		Point2f mp = midPoint(markerCorners[i][0], markerCorners[i][1]);
		Point2f mp2 = midPoint(markerCorners[i][1], markerCorners[i][2]);
		float theta = atan2(center.y - mp.y, center.x - mp.x);
		theta = theta * 180.0 / PI;
		theta = 180 - theta;
		if(markerIds[i] < ENTITY_STARTING_INDEX && markerIds[i] >= 0){
			Robot robot(markerIds[i]);
			robot.set_x(center.x);
			robot.set_y(center.y);
			robot.set_theta(theta);
			robot.set_width(2.0 * dist2pf(mp, center));
			robot.set_height(2.0 * dist2pf(mp2, center));
			robots.push_back(robot);
		} else if(markerIds[i] >= ENTITY_STARTING_INDEX ){
			Entity entity(markerIds[i]);
			entity.set_x(center.x);
			entity.set_y(center.y);
			entity.set_theta(theta);
			entity.set_width(2.0 * dist2pf(mp, center));
			entity.set_height(2.0 * dist2pf(mp2, center));
			entities.push_back(entity);
		}
	}
    updateField();
    applyPerspectiveToObjects();
	mtx.unlock();
}

void printUsage() {
    stringstream message;
    message << "Usage: rbe_server [OPTION...]" << endl << endl;
    message << "  -p\t\tspecify the port for server to run on (default 80)" << endl;
    message << "  -c\t\tspecify the index of the camera to use" << endl;
    message << "  -h\t\tdisplay this help message" << endl;
    message << "  -i\t\tspecify the path to an image to use" << endl << endl;
    message << "One and only one of -c or -i must be specified. If multiple options are passed, the first one wins." << endl;
    cout << message.str();
}

bool parseOptions(int argc, char* argv[], int& errCode, Mat& inputImage, VideoCapture& cap) {
    bool valid = false;
    for (size_t i = 1; i < argc; i++) {
        string opt = string(argv[i]);
        if (opt == "-p") {
            if (i + 1 >= argc) {
                cerr << "Option requires an argument" << endl;
                printUsage();
                errCode = 5;
                return false;
            }
            string port = string(argv[i + 1]);
            blue.open(port);
            i++;
        } else if (opt == "-c") {
            if (valid) {
                i++;
                continue;
            }
            if (i + 1 >= argc) {
                cerr << "Option requires an argument" << endl;
                printUsage();
                errCode = 5;
                return false;
            }
            int camera_index = atoi(argv[i + 1]);
            cout << "Opening Camera" << endl;
            cap.open(camera_index);
            if(!cap.isOpened()){
                cerr << "Error: Could not open camera" << endl;
                printUsage();
                errCode = 2;
                return false;
            }
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
            cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
            cout << "Camera opened" << endl;
            valid = true;
            i++;
        } else if (opt == "-i") {
            if (valid) {
                i++;
                continue;
            }
            if (i + 1 >= argc) {
                cerr << "Option requires an argument" << endl;
                printUsage();
                errCode = 5;
                return false;
            }
            string imagePath = string(argv[i + 1]);
            Mat image = imread(imagePath, IMREAD_COLOR);
            if (!image.data) {
                cerr << "Could not read image data" << endl;
                printUsage();
                errCode = 3;
                return false;
            }
            inputImage = image.clone();
            valid = true;
            i++;
        } else if (opt == "-h") {
            printUsage();
            errCode = 0;
            return false;
        }
    }

    if (!valid) {
        printUsage();
        errCode = 5;
        return false;
    }
    errCode = 0;
    return true;
}

int main(int argc, char* argv[]){
    if(argc < 2) {
        printUsage();
        return 1;
    };

    VideoCapture cap;
    Mat sourceImage;
    Mat inputImage;
    int errCode;
    if (!parseOptions(argc, argv, errCode, sourceImage, cap)) {
        return errCode;
    }

    cout << "Launching network thread" << endl;
    thread network_thread(network, std::ref(robots), std::ref(entities));
    cout << "Detaching thread" << endl;
    network_thread.detach();


	while(true){
        if (cap.isOpened()) {
            cap >> inputImage;
        } else {
            inputImage = sourceImage.clone();
        }
		if(!inputImage.empty()){
			vector<int> markerIds;
			vector< vector<Point2f> > markerCorners;
			runAruco(inputImage, markerIds, markerCorners);
			thread update_thread(update, markerIds, markerCorners);
			update_thread.detach();
			aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);
            if (_drawGrid) {
                applyPerspective(inputImage);
                drawGrid(inputImage);
            }
			imshow("Tracking Vision", inputImage);
			if(waitKey(1) == 27){break;}
		}
	}
	return 0;
}
