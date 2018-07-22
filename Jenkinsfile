pipeline {
    agent {
        docker {
            image 'python:3.5.1'
        }
    }
    stages {
        stage('BUILD') {
            steps {
                sh 'cd python'
                sh 'python setup.py install'
            }
        }
        stage('TEST') {
            steps {
                sh 'python tests/test_basics.py'
            }
        }
    }
}
