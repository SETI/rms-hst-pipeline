{-# LANGUAGE ScopedTypeVariables #-}
-- Modeling the Python 'ProductPass' module.
module ProductPass where

import Control.Exception
import Control.Monad

-- IO' is exception-free IO
type IO' = IO

{-
instance Functor IO' where
    fmap = undefined
instance Applicative IO' where
    pure = undefined
    (<*>) = undefined
instance Monad IO' where
    (>>=) = undefined
-}

runIO' :: IO' a -> IO a
runIO' = id

type H a = IO' (Result a)

runH :: H a -> IO (Result a)
runH = runIO'

-- FITS stuff
data HDU

header :: HDU -> Header
header = undefined

data' :: HDU -> Data
data' = undefined

data Header
data Data
data Fits

fitsOpen :: FilePath -> IO Fits
fitsOpen = undefined

fitsClose :: Fits -> IO ()
fitsClose = undefined

enumerateHDUs :: Fits -> [(Int, HDU)]
enumerateHDUs = undefined

-- PDART stuff
data Product
data File

fullFilepath :: File -> FilePath
fullFilepath = undefined

files :: Product -> [File]
files = undefined

-- Heuristic stuff

data Result a

isFailure :: Result a -> Bool
isFailure = undefined

isSuccess :: Result a -> Bool
isSuccess = undefined

value :: Result a -> a
value = undefined

liftH1 :: (a -> IO b) -> Result a -> H b
liftH1 = undefined

liftH2 :: (a -> b -> IO c) -> Result a -> Result b -> H c
liftH2 = undefined

(<<*) :: H a -> H b -> H a
(<<*) a b = do
    r <- a
    _ <- b
    return r

andThen :: (a -> H b) -> (b -> H c) -> a -> H c
andThen = undefined

instance Functor Result where
    fmap = undefined
instance Applicative Result where
    pure = undefined
    (<*>) = undefined
instance Monad Result where
    (>>=) = undefined

toHeuristic :: IO a -> H a
toHeuristic = undefined

sequence' :: [Result a] -> Result [a]
sequence' = undefined

-- ProductPass

data ProductPass prod file hdu hdr dat = ProductPass {
    processProduct :: Product -> [file] -> IO prod,
    processFile :: File -> [hdu] -> IO file,
    processHdu :: Int -> HDU -> hdr -> dat -> IO hdu,
    processHduHeader :: Header -> IO hdr,
    processHduData :: Data -> IO dat
    }

runProduct :: forall prod file hdu hdr dat
           . ProductPass prod file hdu hdr dat -> Product -> H prod
runProduct productPass product = (doFiles `andThen` doProd) product
    where
    doFiles :: Product -> H [file]
    doFiles prod = do
        mRes <- sequence [runFile productPass f | f <- files prod]
        return $ sequence' mRes

    doProd :: [file] -> H prod
    doProd = toHeuristic . processProduct productPass product
    

-- returns only first exception, like Either
runHDU :: forall prod file hdu hdr dat
       . ProductPass prod file hdu hdr dat -> Int -> HDU -> H hdu
runHDU productPass n hdu = toHeuristic $ do
    mHdr <- processHduHeader productPass $ header hdu
    mDat <- processHduData productPass $ data' hdu
    processHdu productPass n hdu mHdr mDat

-- returns multiple exceptions
runHDU' :: forall prod file hdu hdr dat
        . ProductPass prod file hdu hdr dat -> Int -> HDU -> H hdu
runHDU' productPass n hdu = do
    mHdr <- toHeuristic $ processHduHeader productPass $ header hdu
    mDat <- toHeuristic $ processHduData productPass $ data' hdu
    liftH2 (processHdu productPass n hdu) mHdr mDat

runFile :: forall prod file hdu hdr dat
        . ProductPass prod file hdu hdr dat -> File -> H file
runFile productPass file = do
    let fp = fullFilepath file
    mHDUs <- doFits productPass fp
    liftH1 (processFile productPass file) mHDUs

doFits :: forall prod file hdu hdr dat
       . ProductPass prod file hdu hdr dat -> FilePath -> H [hdu]
doFits productPass
    = (toHeuristic . fitsOpen) `andThen` fitsToResults

    where
    fitsToResults :: Fits -> H [hdu]
    fitsToResults fits = do
        let nHdus = enumerateHDUs fits
        results <- sequence [runHDU productPass n hdu | (n, hdu) <- nHdus]
        _ <- toHeuristic $ fitsClose fits
        return $ sequence' results

{-
finally :: IO a -> IO b -> IO a
finally action cleanup = undefined
-}